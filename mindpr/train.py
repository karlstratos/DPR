import argparse
import os


def main(args):
    import torch
    import transformers

    from copy import deepcopy
    from data import DPRDataset, get_loaders
    from datetime import datetime
    from model import init_train_components, get_loss, validate_by_rank
    from torch.cuda.amp import GradScaler
    from torch.distributed import init_process_group
    from transformers import AutoTokenizer, set_seed
    from util import Logger, check_distributed, strtime

    # transformers.logging.set_verbosity_error()  # Not needed in this version

    set_seed(args.seed)
    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1

    logger = Logger(log_path=args.model + '.log', on=is_main_process)
    logger.log(str(args))
    logger.log(f'rank {rank} local_rank {local_rank} world_size {world_size}',
               force=True)

    if is_distributed:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        init_process_group('nccl')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.log(f'Using device: {str(device)}', force=True)

    dataset_train = DPRDataset(args.data_train)
    dataset_val = DPRDataset(args.data_val)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    loader_train, loader_val = get_loaders(dataset_train, dataset_val,
                                           tokenizer, args, rank, world_size)
    num_training_steps = len(loader_train) * args.epochs
    effective_batch_size = args.batch_size
    if world_size > 0:
        effective_batch_size *= world_size
    logger.log(f'INFO\n'
               f'  - # train examples:     {len(dataset_train)}\n'
               f'  - # val examples:       {len(dataset_val)}\n'
               f'  - batch size:           {min(args.batch_size, len(dataset_train))}\n'
               f'  - batch size val:       {min(args.batch_size_val, len(dataset_val))}\n'
               f'  - # batches:            {len(loader_train)}\n'
               f'  - # batches val:        {len(loader_val)}\n'
               f'  - # warmup steps:       {args.num_warmup_steps}\n'
               f'  - # epochs:             {args.epochs}\n'
               f'  - effective batch size: {effective_batch_size}')

    model, optimizer, scheduler = init_train_components(tokenizer, args.lr,
                                                        args.num_warmup_steps,
                                                        num_training_steps,
                                                        dropout=args.dropout,
                                                        device=device)

    logger.log(f'Optimizer\n{str(optimizer)}')
    logger.log(f'\nQuery encoder (passage encoder same independent)')
    logger.log(f'{str(model.query_encoder.encoder.config)}')

    if is_distributed:
        logger.log('DDP wrapping')

        # Without setting find_unused_parameters=True we get a runtime error:
        # https://discuss.pytorch.org/t/how-to-find-the-unused-parameters-in-network/63948/4
        # https://github.com/facebookresearch/DPR/blob/1ee31c6c5335f05c45138161eb5ce2c03023115d/dpr/utils/model_utils.py#L63
        # TODO: I changed add_pooling_layer to False, now I get a warning.
        # Try setting it to False again.
        # Yes, False gets rid of the warning.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
            #find_unused_parameters=False)
    else:
        logger.log('Single-process single-device, no model wrapping')

    # Training
    scaler = GradScaler()
    def scale_grad(model, C):
        for p in model.parameters():
            # grad is sometimes None for BERT, which is probably why we get that
            # DDP unused parameters error.
            if p.grad is not None:
                p.grad *= C

    num_batches_processed = 0
    num_steps_skipped = 0
    loss_val_best = float('inf')
    sd_best = None
    start_time = datetime.now()
    for epoch in range(1, args.epochs + 1):
        model.train()
        num_steps_per_epoch = 0
        loss_sum = 0.
        num_correct_sum = 0

        for batch in loader_train:
            loss, num_correct = get_loss(model, batch, rank, world_size,
                                         args.autocast, device)
            loss_sum += loss.item()
            num_correct_sum += num_correct.item()
            num_batches_processed += 1

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if is_distributed:
                scale_grad(model, world_size)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()
            skipped_step = (scaler.get_scale() < scale)
            if skipped_step:
                num_steps_skipped += 1
            else:
                scheduler.step()
                num_steps_per_epoch += 1
            optimizer.zero_grad()

        loss_train = loss_sum / len(loader_train)
        acc = num_correct_sum / len(dataset_train) * 100.
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        loss_val = validate_by_rank(model, loader_val, rank, world_size,
                                    args.autocast, device)
        is_best_string = ''
        if loss_val < loss_val_best:
            sd = model.module.state_dict() if is_distributed else \
                 model.state_dict()
            sd_best = deepcopy(sd)
            is_best_string = ' <-------------'
            loss_val_best = loss_val   # Sketchy

        logger.log(f'End of epoch {epoch:3d}: {num_steps_per_epoch} steps '
                   f'({num_steps_skipped} skipped so far), per-batch loss '
                   f'{loss_train:10.4f}, acc {acc:10.2f}, lr '
                   f' {current_lr:.2E}, loss val {loss_val:10.4f} '
                   f'{is_best_string}')

    if is_main_process and sd_best is not None:
        logger.log(f'\nDone training | total time {strtime(start_time)} | '
                   f'saving best model to {args.model}')
        torch.save({'sd': sd_best, 'args': args}, args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('data_train', type=str)
    parser.add_argument('data_val', type=str)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--batch_size_val', type=int, default=10)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_warmup_steps', type=int, default=40)
    parser.add_argument('--clip', type=float, default=2.)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--pad_to_max', action='store_true')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
