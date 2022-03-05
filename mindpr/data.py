import csv
import random
import torch

from file_handling import read_json
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class DPRDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        extension = Path(path).suffix
        if extension == '.json':
            self.samples = read_json(path)
        elif extension == '.csv':
            self.samples = []
            with open(path) as f:
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    self.samples.append({'question': row[0],
                                         'answers': eval(row[1])})
        else:
            raise ValueError('Invalid DPR file extension: ' + extension)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def text2tensor(tokenizer, inputs, titles=None, max_length=256):
    if titles is not None:
        tensor = tokenizer(text=titles, text_pair=inputs, padding='max_length',
                           truncation=True, max_length=max_length,
                           return_tensors='pt')
    else:
        tensor = tokenizer(text=inputs, padding='max_length', truncation=True,
                           max_length=max_length, return_tensors='pt')

    tensor['input_ids'][:, -1] = tokenizer.sep_token_id
    tensor['attention_mask'][:, -1] = 1
    tensor['token_type_ids'].fill_(0)
    return tensor


def tensorize(samples, tokenizer, max_length, shuffle=False,
              num_hard_negatives=1, num_other_negatives=0):
    queries = []
    labels = []
    titles = []
    texts = []

    for sample in samples:
        if shuffle:
            random.shuffle(sample['negative_ctxs'])
            random.shuffle(sample['hard_negative_ctxs'])

        queries.append(sample['question'])
        labels.append(len(titles))

        titles.append(sample['positive_ctxs'][0]['title'])
        texts.append(sample['positive_ctxs'][0]['text'])

        other_negs = sample['negative_ctxs'][:num_other_negatives]
        titles.extend([other_neg['title'] for other_neg in other_negs])
        texts.extend([other_neg['text'] for other_neg in other_negs])

        # Doing what DPR does, but this seems wrong because if hard negs are
        # used but fall back to other negs, AND other negs are used, they're
        # repeated.
        if sample['hard_negative_ctxs']:
            hard_negs = sample['hard_negative_ctxs']
        else:
            hard_negs = sample['negative_ctxs']
        hard_negs = hard_negs[:num_hard_negatives]
        titles.extend([hard_neg['title'] for hard_neg in hard_negs])
        texts.extend([hard_neg['text'] for hard_neg in hard_negs])

    queries = text2tensor(tokenizer, queries, max_length=max_length)

    #for text in texts:
    #    print(text[:100])
    #print()
    #for title in titles:
    #    print(title)
    passages = text2tensor(tokenizer, texts, titles=titles,
                           max_length=max_length)

    Q = queries['input_ids']  # (B, L)
    Q_mask = queries['attention_mask']
    Q_type = queries['token_type_ids']
    P = passages['input_ids']  # (MB, L)
    P_mask = passages['attention_mask']
    P_type = passages['token_type_ids']
    labels = torch.LongTensor(labels)  # (B,) elements in [0, MB)

    return Q, Q_mask, Q_type, P, P_mask, P_type, labels


def get_loaders(dataset_train, dataset_val, tokenizer, args, rank=-1,
                world_size=-1):
    collate_fn = lambda samples: tensorize(samples, tokenizer, args.max_length,
                                           shuffle=not args.no_shuffle)
    collate_fn_val = lambda samples: tensorize(samples, tokenizer,
                                               args.max_length,
                                               shuffle=not args.no_shuffle,
                                               num_hard_negatives=30,
                                               num_other_negatives=30)

    if world_size != -1:
        def make_distributed_loader(dataset, batch_size, shuffle, collate_fn):
            sampler = DistributedSampler(dataset, num_replicas=world_size,
                                         rank=rank, shuffle=shuffle,
                                         seed=args.seed, drop_last=False)
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=collate_fn, sampler=sampler)
            return loader

        loader_train = make_distributed_loader(dataset_train, args.batch_size,
                                               not args.no_shuffle, collate_fn)
        loader_val = make_distributed_loader(dataset_val, args.batch_size_val,
                                             False, collate_fn_val)
    else:

        def make_regular_loader(dataset, batch_size, shuffle, collate_fn):
            loader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=args.num_workers,
                                collate_fn=collate_fn)
            return loader

        loader_train = make_regular_loader(dataset_train, args.batch_size,
                                           not args.no_shuffle, collate_fn)
        loader_val = make_regular_loader(dataset_val, args.batch_size_val,
                                         False, collate_fn_val)

    return loader_train, loader_val


def get_loader_passages(dataset, collate_fn, args, rank=-1, world_size=-1):
    if world_size != -1:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=False, drop_last=False)
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    return loader


def tensorize_questions(samples, tokenizer, max_length):
    queries = []

    for sample in samples:
        queries.append(sample['question'])

    # [CLS] query [SEP]
    queries = text2tensor(tokenizer, queries, max_length=max_length)
    Q = queries['input_ids']  # (B, L)
    Q_mask = queries['attention_mask']
    Q_type = queries['token_type_ids']

    return Q, Q_mask, Q_type


class WikiPassageDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        self.samples = []
        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if row[0] == 'id':
                    continue
                self.samples.append({'pid': int(row[0]), 'text': row[1],
                                     'title': row[2]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def tensorize_passages(samples, tokenizer, max_length):
    titles = []
    texts = []
    pids = []

    for sample in samples:
        titles.append(sample['title'])
        texts.append(sample['text'])
        pids.append(sample['pid'])

    # [CLS] title [SEP] text [SEP]
    passages = text2tensor(tokenizer, texts, titles=titles,
                           max_length=max_length)

    P = passages['input_ids']  # (B, L)
    P_mask = passages['attention_mask']
    P_type = passages['token_type_ids']
    I = torch.LongTensor(pids)  # Need to save pids for distributed

    return P, P_mask, P_type, I
