import json
import pickle
import random
import os

import itertools
import math
import torch
from typing import List, Iterator, Callable, Tuple
from biencoder_data import JsonQADataset


def my_get_iterator(
        dataset_path,
        batch_size: int,
        shuffle=True,
        shuffle_seed: int = 0,
        rank: int = 0,
        world_size: int = -9,  # My addition
):
    dataset = JsonQADataset(dataset_path)  # So this must be correct
    sharded_iterator = ShardedDataIterator(
        dataset,
        shard_id=rank,
        num_shards=world_size,
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
    )

    return MultiSetDataIterator(
        [sharded_iterator],
        shuffle_seed,
        shuffle,
    )



class ShardedDataIterator:

    def __init__(
        self,
        data: torch.utils.data.Dataset,
        shard_id: int = 0,  # local rank
        num_shards: int = 1,  # num processes
        batch_size: int = 1,
        shuffle=True,
        shuffle_seed: int = 0,
    ):
        self.data = data
        total_size = len(data)
        self.shards_num = max(num_shards, 1)
        self.shard_id = max(shard_id, 0)

        # Num samples per process: ceiling makes sure the last process gets fewer examples unless # examples is divisible by # processes.
        samples_per_shard = math.ceil(total_size / self.shards_num)
        self.shard_start_idx = self.shard_id * samples_per_shard

        # We do NOT ditch the trailing data subset. If 11 examples and 3 processes, 2 processes will be
        # assigned 4 examples each and 1 process will be assigned 3 examples.
        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        # In each process, the number of iterations is the same even if it has fewer examples.
        self.max_iterations = int(samples_per_shard / batch_size)
        #-------------------------------------------------------
        # Example (this is verified by running the command
        #
        # torchrun --standalone --nnodes=1 --nproc_per_node=3 mindpr/train.py /tmp/model downloads/data/retriever/nq-train11.json downloads/data/retriever/nq-train11.json --batch_size 2 --batch_size_val 1 --num_warmup_steps 1 --lr 1e-4 --epochs 1 --gpus 0,1,2 --start_epoch_val 0 --log_result_step 1 --epochs 2
        #
        #): 11 examples, 3 processes, batch size 2. Thus samples_per_shard = ceil(11/3) = 4 (but the last process only gets 3 examples) and every process gets max_iterations = 4/2 = 2. Here's what happens:
        #
        # In epoch 0
        #    shard_id: 0 takes care of indices [7, 10, 9, 3], max_iterations 2
        #    shard_id: 1 takes care of indices [8, 2, 1, 5], max_iterations 2
        #    shard_id: 2 takes care of indices [*4*, 0, 6], max_iterations 2
        #
        #    rank 0 computing indices [7, 10] in batch num 0
        #    rank 1 computing indices [8, 2] in batch num 0
        #    rank 2 computing indices [*4*, 0] in batch num 0
        #
        #    rank 0 computing indices [9, 3] in batch num 1
        #    rank 1 computing indices [1, 5] in batch num 1
        #    rank 2 computing indices [6, *4*] in batch num 1   ** Note that the process that has a smaller subset uses the first (shuffled) indices WITHIN THE ASSIGNMENT (thus 4) to make the batch size 2
        #
        # In epoch 1
        #    shard_id: 0 takes care of indices [0, 3, 8, 4], max_iterations 2
        #    shard_id: 1 takes care of indices [2, 9, 5, 1], max_iterations 2
        #    shard_id: 2 takes care of indices [*6*, 10, 7], max_iterations 2
        #
        #    rank 0 computing indices [0, 3] in batch num 0
        #    rank 1 computing indices [2, 9] in batch num 0
        #    rank 2 computing indices [*6*, 10] in batch num 0
        #
        #    rank 0 computing indices [8, 4] in batch num 1
        #    rank 1 computing indices [5, 1] in batch num 1
        #    rank 2 computing indices [7, *6*] in batch num 1   ** Note that the process that has a smaller subset uses the first (shuffled) indices WITHIN THE ASSIGNMENT (thus 6) to make the batch size 2
        #
        # In contrast, if I use DistributedSampler and DataLoader all with drop_last=False (above command with --use_my_loader), this happens:
        #
        # In epoch 0
        #    Shuffle 11 indices: [*8*, 4, 2, 1, 5, 6, 3, 7, 9, 10, 0]
        #
        #    rank 0 computing indices [*8*, 4] in batch num 0
        #    rank 1 computing indices [2, 1] in batch num 0
        #    rank 2 computing indices [5, 6] in batch num 0
        #
        #    rank 0 computing indices [3, 7] in batch num 1
        #    rank 1 computing indices [9, 10] in batch num 1
        #    rank 2 computing indices [0, *8*] in batch num 1   ** The last process uses  the first (shuffled) indices in THE WHOLE DATA (thus 8) to make the batch size 2, they come from the first process
        #
        # In epoch 1
        #    Shuffle 11 indices: [*6*, 7, 3, 5, 10, 2, 1, 8, 9, 4, 0]
        #
        #    rank 0 computing indices [*6*, 7] in batch num 0
        #    rank 1 computing indices [3, 5] in batch num 0
        #    rank 2 computing indices [10, 2] in batch num 0
        #
        #    rank 0 computing indices [1, 8] in batch num 1
        #    rank 1 computing indices [9, 4] in batch num 1
        #    rank 2 computing indices [0, *6*] in batch num 1   ** The last process uses  the first (shuffled) indices in THE WHOLE DATA (thus 6) to make the batch size 2, they come from the first process

        self.iteration = 0
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed

    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.data)))
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices[self.shard_start_idx : self.shard_end_idx]
        print(f'shard_id: {self.shard_id} takes care of indices {str(shard_indices)}, max_iterations {self.max_iterations}')

        return shard_indices  # This is just a shuffled list of item indices

    def iterate_ds_sampled_data(self, num_iterations: int, epoch: int = 0) -> Iterator[List]:
        self.iteration = 0
        shard_indices = self.get_shard_indices(epoch)  # Shuffled items (all)
        cycle_it = itertools.cycle(shard_indices)
        for i in range(num_iterations):
            items_idxs = [next(cycle_it) for _ in range(self.batch_size)]
            self.iteration += 1
            items = [self.data[idx] for idx in items_idxs]
            yield items, items_idxs

        self.iteration = 0


class MultiSetDataIterator:

    def __init__(
        self,
        datasets: List[ShardedDataIterator],
        shuffle_seed: int = 0,
        shuffle=True,
    ):
        self.iterables = datasets
        data_lengths = [len(it.data) for it in datasets]
        self.total_data = sum(data_lengths)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0

        self.max_its_pr_ds = [ds.max_iterations for ds in datasets]

        self.max_iterations = sum(self.max_its_pr_ds)

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[Tuple[List, int]]:

        data_src_indices = []  # will always be [0] if using single dataset
        iterators = []
        for source, src_its in enumerate(self.max_its_pr_ds):
            data_src_indices.extend([source] * src_its)

            iterators.append(self.iterables[source].iterate_ds_sampled_data(src_its, epoch=epoch))

        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(data_src_indices)


        for i, source_idx in enumerate(data_src_indices):
            it = iterators[source_idx]
            next_item = next(it, None)  # List of BiEncoder samples
            if next_item is not None:

                self.iteration += 1
                yield (next_item, source_idx)  # ((items, indices), source_idx)

        [next(it, None) for it in iterators]

        # TODO: clear iterators in some non-hacky way
        for it in self.iterables:
            it.iteration = 0
        # reset the iteration status
        self.iteration = 0
