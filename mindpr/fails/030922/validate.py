# Debugging validation
import argparse
import os


def main(args):
    import torch
    import transformers

    from data import DPRDataset, get_loader_val
    from datetime import datetime
    from model import init_train_components, validate_by_rank, \
        validate_by_rank_naive
    from torch.distributed import init_process_group
    from transformers import AutoTokenizer, set_seed
    from util import Logger, check_distributed, strtime

    rank, local_rank, world_size = check_distributed()
    is_main_process = local_rank in [-1, 0]
    is_distributed = world_size != -1

    logger = Logger(on=is_main_process)
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

    dataset_val = DPRDataset(args.data_val)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    loader_val = get_loader_val(dataset_val, tokenizer, args, rank, world_size,
                                num_hard_negatives=args.num_hard_negatives,
                                num_other_negatives=args.num_other_negatives)
    model, _, _ = init_train_components(tokenizer, 42, 42, 42, device=device)

    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
            #find_unused_parameters=False)
    else:
        logger.log('Single-process single-device, no model wrapping')

    model.eval()
    logger.log('validate_by_rank')
    start_time = datetime.now()
    avgrank, num_cands_avg = validate_by_rank(model, loader_val, rank,
                                              world_size, args.autocast, device,
                                              args.subbatch_size,
                                              disable_tqdm=False)
    logger.log(f'\nDone | On average, rank {avgrank:4.3f} out of '
               f'{num_cands_avg:4.3f} cands per process | '
               f'{strtime(start_time)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_val', type=str, default='/common/home/jl2529/repositories/DPR/downloads/data/retriever/nq-dev.json')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size_val', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_hard_negatives', type=int, default=30)
    parser.add_argument('--num_other_negatives', type=int, default=30)
    parser.add_argument('--subbatch_size', type=int, default=128)
    parser.add_argument('--autocast', action='store_true')
    parser.add_argument('--gpus', default='', type=str)
    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
