import collections
import glob
import json
from typing import List

import numpy as np
import torch


BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])


class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]


class JsonQADataset(torch.utils.data.Dataset):
    def __init__(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __getitem__(self, index) -> BiEncoderSample:
        json_sample = self.data[index]
        r = BiEncoderSample()
        r.query = json_sample["question"]

        positive_ctxs = json_sample["positive_ctxs"]
        negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        r.positive_passages = [BiEncoderPassage(ctx["text"], ctx["title"]) for ctx in positive_ctxs]
        r.negative_passages = [BiEncoderPassage(ctx["text"], ctx["title"]) for ctx in negative_ctxs]
        r.hard_negative_passages = [BiEncoderPassage(ctx["text"], ctx["title"]) for ctx in hard_negative_ctxs]
        return r

    def __len__(self):
        return len(self.data)
