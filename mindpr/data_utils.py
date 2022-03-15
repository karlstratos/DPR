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
        rank=rank,
    )



class ShardedDataIterator(object):

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

        samples_per_shard = math.ceil(total_size / self.shards_num)  # Num samples per GPU

        self.shard_start_idx = self.shard_id * samples_per_shard

        # Note we ditch the trailing data subset???
        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        # In this shard, we ditch the trailing batch.
        self.max_iterations = int(samples_per_shard / batch_size)
        #-------------------------------------------------------
        # This happens for each PROCESS, and each process is assigned a nonoverlapping subset of data.
        # Example: num examples 10, batch size 4, 2 processes might be assigned
        #    Process 1: [8, 1, 3, 4]
        #    Process 2: [9, 7, 0, 2]
        # and an epoch will involve **1** gradient step on the 8 examples
        #string = f'total_size: {total_size}, shards_num: {self.shards_num}, shard_id: {self.shard_id}, shard_start_idx: {self.shard_start_idx}, shard_end_idx: {self.shard_end_idx}, batch_size: {batch_size}, max_iterations: {self.max_iterations}'
        #print(string)

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
        return shard_indices  # This is just a shuffled list of item indices

    def iterate_ds_sampled_data(self, num_iterations: int, epoch: int = 0) -> Iterator[List]:
        self.iteration = 0
        shard_indices = self.get_shard_indices(epoch)  # Shuffled items (all)
        cycle_it = itertools.cycle(shard_indices)
        for i in range(num_iterations):  # Note num_iterations: it'll skip the trailing batch
            items_idxs = [next(cycle_it) for _ in range(self.batch_size)]
            self.iteration += 1
            items = [self.data[idx] for idx in items_idxs]
            yield items, items_idxs

        self.iteration = 0


class MultiSetDataIterator(object):
    """
    Iterator over multiple data sources. Useful when all samples form a single batch should be from the same dataset.
    """

    def __init__(
        self,
        datasets: List[ShardedDataIterator],
        shuffle_seed: int = 0,
        shuffle=True,
        rank: int = 0,
    ):
        self.iterables = datasets
        data_lengths = [len(it.data) for it in datasets]
        self.total_data = sum(data_lengths)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0
        self.rank = rank

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
