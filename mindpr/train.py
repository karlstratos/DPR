import argparse
import os


def main(args):
    import torch
    import transformers

    from copy import deepcopy
    from data import DPRDataset, get_loaders
    from datetime import datetime
    from model import init_train_components, get_loss, validate_by_rank
    from torch.distributed import init_process_group
    from transformers import AutoTokenizer, set_seed
    from util import Logger, check_distributed, strtime

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
                                           tokenizer, args, rank, world_size,
                                           args.num_hard_negatives_val,
                                           args.num_other_negatives_val)
    num_training_steps = len(loader_train) * args.epochs
    effective_batch_size = args.batch_size
    if world_size > 0:
        effective_batch_size *= world_size

    model, optimizer, scheduler = init_train_components(tokenizer, args.lr,
                                                        args.num_warmup_steps,
                                                        num_training_steps,
                                                        dropout=args.dropout,
                                                        device=device)

    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    else:
        logger.log('Single-process single-device, no model wrapping')

    # Training
    num_batches_processed = 0
    num_steps_skipped = 0
    loss_val_best = float('inf')
    sd_best = None
    start_time = datetime.now()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.
        num_correct_sum = 0

        for batch_num, batch in enumerate(loader_train):
            loss, num_correct = get_loss(model, batch, rank, world_size, device)
            loss_sum += loss.item()
            num_correct_sum += num_correct.item()
            num_batches_processed += 1

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if batch_num % args.log_result_step == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.log(f'Epoch {epoch}, step {batch_num + 1}/{len(loader_train)}, loss {loss.item():4.3f}, lr {lr:f}')

        loss_train = loss_sum / len(loader_train)
        acc = num_correct_sum / len(dataset_train) * 100.

        loss_val = -1
        is_best_string = ''
        if epoch >= args.start_epoch_val:
            model.eval()
            avgrank, num_cands = validate_by_rank(model, loader_val, rank,
                                                  world_size, device,
                                                  args.subbatch_size)
            loss_val = avgrank
            if loss_val < loss_val_best:
                sd = model.module.state_dict() if is_distributed else \
                     model.state_dict()
                sd_best = deepcopy(sd)
                is_best_string = ' <-------------'
                loss_val_best = loss_val

        #logger.log(f'Epoch {epoch:3d}: per-batch loss {loss_train:10.4f}, acc {acc:10.2f}, val {loss_val:4.3f} {is_best_string}')

    if is_main_process and sd_best is not None:
        logger.log(f'\nDone training | total time {strtime(start_time)} | '
                   f'saving best model to {args.model}')
        torch.save({'sd': sd_best, 'args': args}, args.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('data_train', type=str)
    parser.add_argument('data_val', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--batch_size_val', type=int, default=512)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_warmup_steps', type=int, default=0)
    parser.add_argument('--clip', type=float, default=2.)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_hard_negatives_val', type=int, default=30)
    parser.add_argument('--num_other_negatives_val', type=int, default=30)
    parser.add_argument('--subbatch_size', type=int, default=1024)
    parser.add_argument('--start_epoch_val', type=int, default=30)
    parser.add_argument('--log_result_step', type=int, default=99999999)
    parser.add_argument('--no_shuffle', action='store_true')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
