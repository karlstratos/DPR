import collections
import glob
import itertools
import json
import math
import os
import pickle
import random
import torch

from typing import List, Iterator, Callable, Tuple


BiEncoderPassage = collections.namedtuple('BiEncoderPassage', ['text', 'title'])


class BiEncoderSample:
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class JsonQADataset(torch.utils.data.Dataset):

    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = json_sample['question']

        positive_ctxs = json_sample['positive_ctxs']
        negative_ctxs = json_sample.get('negative_ctxs', [])
        hard_negative_ctxs = json_sample.get('hard_negative_ctxs', [])

        r.positive_passages = [BiEncoderPassage(ctx['text'], ctx.get('title')) for ctx in positive_ctxs]
        r.negative_passages = [BiEncoderPassage(ctx['text'], ctx.get('title')) for ctx in negative_ctxs]
        r.hard_negative_passages = [BiEncoderPassage(ctx['text'], ctx.get('title')) for ctx in hard_negative_ctxs]
        return r


def shuffle_with_seed(x: List, seed):
    rnd = random.Random(seed)
    rnd.shuffle(x)


class DatasetSharder:

    def __init__(self, dataset, shard_id=0, num_shards=1, batch_size=1, shuffle=True, shuffle_seed=0):
        self.dataset = dataset
        self.shard_id = max(shard_id, 0)  # -1 => 0
        self.num_shards = max(num_shards, 1)  # -1 => 1

        num_samples_per_shard = math.ceil(len(dataset) / self.num_shards)
        self.shard_start_idx = self.shard_id * num_samples_per_shard
        self.shard_end_idx = min(self.shard_start_idx + num_samples_per_shard, len(dataset))  # Last shard may get fewer indices
        self.num_iterations = int(num_samples_per_shard / batch_size)  # But it will make the same number of iterations (round-robin style)

        self.iteration = 0
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed

    def get_shard_indices(self, epoch):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            shuffle_with_seed(indices, self.shuffle_seed + epoch)  # Same shuffling in same epoch
        return indices[self.shard_start_idx : self.shard_end_idx]

    def get_iterator(self, epoch, num_iterations=None) -> Iterator[Tuple[List, List]]:
        shard_indices = self.get_shard_indices(epoch)
        cycle_iter = itertools.cycle(shard_indices)
        self.iteration = 0
        for _ in range(self.num_iterations if num_iterations is None else num_iterations):
            items_idxs = [next(cycle_iter) for _ in range(self.batch_size)]  # This will pluck the same batch size round-robin
            items = [self.dataset[idx] for idx in items_idxs]
            self.iteration += 1
            yield items, items_idxs


class MultiDatasetSharder:

    def __init__(self, sharders: List[DatasetSharder], shuffle=True, shuffle_seed=0):
        self.sharders = sharders
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.num_iterations = sum(sharder.num_iterations for sharder in sharders)
        self.iteration = 0

    def get_iterator(self, epoch) -> Iterator[Tuple[Tuple[List, List], int]]:
        sharder_indices = []
        iterators = []

        for sharder_index, sharder in enumerate(self.sharders):
            sharder_indices.extend([sharder_index] * sharder.num_iterations)
            iterators.append(sharder.get_iterator(epoch))

        if self.shuffle:
            shuffle_with_seed(sharder_indices, self.shuffle_seed + epoch)  # Same shuffling in same epoch

        self.iteration = 0
        for sharder_index in sharder_indices:
            self.iteration += 1
            yield next(iterators[sharder_index]), sharder_index   # (items, indices), sharder_index)


def get_sharder(dataset_paths, batch_size, shuffle=True, shuffle_seed=0, rank=-1, world_size=-1):
    sharders = []
    for dataset_path in dataset_paths:
        dataset = JsonQADataset(dataset_path)
        sharder = DatasetSharder(dataset, rank, world_size, batch_size, shuffle, shuffle_seed)
        sharders.append(sharder)
    return MultiDatasetSharder(sharders, shuffle, shuffle_seed)
