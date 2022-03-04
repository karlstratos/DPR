import argparse
import os


def main(args):
    import glob
    import pickle
    import torch
    import torch.distributed as dist
    import transformers

    from data import WikiPassageDataset, tensorize_passages, get_loader_passages
    from datetime import datetime
    from file_handling import mkdir_optional
    from model import load_model, run_forward_encode
    from pathlib import Path
    from torch.cuda.amp import autocast
    from torch.distributed import init_process_group
    from transformers import AutoTokenizer, set_seed
    from tqdm import tqdm
    from util import Logger, check_distributed, strtime

    # transformers.logging.set_verbosity_error()  # Not needed in this version

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

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model, args_saved = load_model(args.model, tokenizer, device)

    if is_distributed:
        logger.log('DDP wrapping')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
    else:
        logger.log('Single-process single-device, no model wrapping')

    effective_batch_size = args.batch_size
    if world_size > 0:
        effective_batch_size *= world_size

    logger.log(f'INFO\n'
               f'  - batch size:           {args.batch_size}\n'
               f'  - world size:           {world_size}\n'
               f'  - effective batch size: {effective_batch_size}')

    passage_files = sorted(glob.glob(args.passages))
    logger.log(f'passage files identified: {str(passage_files)}')
    start_time = datetime.now()
    model.eval()

    mkdir_optional(args.outdir)
    collate_fn = lambda samples: tensorize_passages(samples, tokenizer,
                                                    args_saved.max_length)
    for passage_file in passage_files:
        dataset_passages = WikiPassageDataset(passage_file)
        logger.log(f'{passage_file} ({len(dataset_passages)} passages)')

        loader = get_loader_passages(dataset_passages, collate_fn, args, rank,
                                     world_size)
        pid_seen = {}
        bucket = []
        with torch.no_grad():
            for batch_num, batch in enumerate(tqdm(loader)):
                with autocast(enabled=args.autocast):
                    Y, I = run_forward_encode(model, batch, world_size, device)
                pid_embedding_pairs = list(zip(I.tolist(), Y.cpu().numpy()))

                # In DDP, last non-full batch is padded with earliest examples
                # (round robin fashion). We need to drop those. It seems there
                # is randomness in order, so we'll play safe and add embeddings
                # only if it's not added already.
                for pid, embedding in pid_embedding_pairs:
                    if pid in pid_seen:
                        continue
                    else:
                        pid_seen[pid] = True
                        bucket.append([pid, embedding])

            # Sort by pid since DDP may have processed items in diff order.
            bucket.sort(key=lambda x: x[0])

            assert len(bucket) == len(dataset_passages)

            # Sanity check
            pids = [int(sample['pid']) for sample in dataset_passages.samples]
            for i, (pid_bucket, _) in enumerate(bucket):
                assert pid_bucket == pids[i]

            if is_main_process and bucket:
                name = Path(passage_file).stem + f'_encoded.pickle'
                path = os.path.join(args.outdir, name)
                with open(path, 'wb') as f:
                    logger.log(f'Dumping {path} (size {len(bucket)})')
                    pickle.dump(bucket, f)


    logger.log(f'\nDone | total time {strtime(start_time)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('passages', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--gpus', default='', type=str)
    parser.add_argument('--autocast', action='store_true')

    args = parser.parse_args()

    # Set environment variables before importing libraries that use them!
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
