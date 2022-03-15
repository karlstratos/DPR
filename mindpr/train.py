import argparse
import os


def main(args):
    import random
    import torch
    import transformers

    from copy import deepcopy
    from data import DPRDataset, get_loaders
    from datetime import datetime
    from model import init_train_components, get_loss, validate_by_rank
    from torch.distributed import init_process_group
    from transformers import AutoTokenizer, set_seed
    from util import Logger, check_distributed, strtime

    # Debuggging with the DPR code
    from biencoder import BiEncoder
    from data_utils import my_get_iterator
    from hf_models import BertTensorizer

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


    iterator_train = my_get_iterator(args.data_train,
                                     args.batch_size,
                                     shuffle=not args.no_shuffle,
                                     shuffle_seed=args.seed,
                                     rank=local_rank,
                                     world_size=world_size)

    dataset_train = DPRDataset(args.data_train)
    dataset_val = DPRDataset(args.data_val)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    loader_train, loader_val = get_loaders(dataset_train, dataset_val, tokenizer, args, rank, world_size, args.num_hard_negatives_val, args.num_other_negatives_val)
    num_training_steps = iterator_train.max_iterations * args.epochs
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

    tensorizer = BertTensorizer(tokenizer, args.max_length)

    # Training
    loss_val_best = float('inf')
    sd_best = None
    start_time = datetime.now()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.
        num_correct_sum = 0
        if is_distributed:
            logger.log(f'Calling loader_train.sampler.set_epoch({epoch}) from rank {rank}', force=True)
            loader_train.sampler.set_epoch(epoch)

        for batch_num, batch in enumerate(loader_train if args.use_my_loader else
                                          iterator_train.iterate_ds_data(epoch=epoch)):  # Drops trailing batch
            if not args.use_my_loader:
                (samples_batch, indices), dataset = batch
                random.seed(args.seed + epoch + iterator_train.iteration)
                biencoder_batch = BiEncoder.create_biencoder_input2(
                    samples_batch,
                    tensorizer,
                    True,
                    1,  # train.hard_negatives
                    0,  # train.other_negatives
                    shuffle=not args.no_shuffle,
                    shuffle_positives=False,
                )
                Q = biencoder_batch.question_ids
                Q_mask = tensorizer.get_attn_mask(Q)
                Q_type = biencoder_batch.question_segments
                P = biencoder_batch.context_ids
                P_mask = tensorizer.get_attn_mask(P)
                P_type = biencoder_batch.ctx_segments
                labels = torch.LongTensor(biencoder_batch.is_positive)
                batch = [Q, Q_mask, Q_type, P, P_mask, P_type, labels, indices]

            #Q, Q_mask, Q_type, P, P_mask, P_type, labels, indices = batch
            #string = f'\nrank={rank}, batch_num={batch_num}'
            #for q in tokenizer.batch_decode(Q, skip_special_tokens=True):
            #    string += '\n'+q
            #string += '\n'+str(Q.size())
            #for p in tokenizer.batch_decode(P, skip_special_tokens=True):
            #    string += '\n'+p[:100]
            #string += '\n'+str(P.size())
            #string += '\n'+str(indices)
            #print(string)

            loss, num_correct = get_loss(model, batch, rank, world_size, device)
            loss_sum += loss.item()
            num_correct_sum += num_correct.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if args.log_result_step != 99999999 and batch_num % args.log_result_step == 0:
                lr = optimizer.param_groups[0]['lr']
                model.eval()
                myval, _ = validate_by_rank(model, loader_val, rank,
                                            world_size, device,
                                            args.subbatch_size)
                model.train()
                logger.log(f'Epoch {epoch}, step {batch_num + 1}/{len(loader_train)}, loss {loss.item():4.6f}, lr {lr:f}, val {myval:4.3f}')

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

        if args.log_result_step == 99999999:
            logger.log(f'Epoch {epoch:3d}: acc {acc:10.2f}, val {loss_val:4.3f} {is_best_string}')

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
    parser.add_argument('--use_my_loader', action='store_true')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
