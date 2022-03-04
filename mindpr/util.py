import os
import sys
import torch
import regex
import unicodedata

from collections import Counter
from datetime import datetime


def contains(lst, sublst):  # True iff lst contains sublst
    for i in range(0, len(lst) - len(sublst) + 1):
        if sublst == lst[i: i + len(sublst)]:
            return True
    return False


def check_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank = -1
        local_rank = -1
        world_size = -1
    return rank, local_rank, world_size


def strtime(datetime_checkpoint):
    diff = datetime.now() - datetime_checkpoint
    return str(diff).rsplit('.')[0]  # Ignore below seconds


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger(object):

    def __init__(self, log_path=None, on=True):
        self.log_path = log_path
        self.on = on

        if self.on and self.log_path is not None:
            while os.path.isfile(self.log_path):
                self.log_path += '+'

    def log(self, string, newline=True, force=False):
        if self.on or force:
            sys.stdout.write(string)
            if newline: sys.stdout.write('\n')
            sys.stdout.flush()

            if self.on and self.log_path is not None:
                with open(self.log_path, 'a') as logf:
                    logf.write(string)
                    if newline: logf.write('\n')
                    logf.flush()


######### BEGIN: Adapted from https://github.com/facebookresearch/DrQA #########

class Tokens:
    """A class to represent a list of tokenized text."""
    TEXT = 0
    TEXT_WS = 1
    SPAN = 2

    def __init__(self, tuples):
        self.tuples = tuples

    def __len__(self):
        return len(self.tuples)

    def words(self, lower=False):
        return [t[self.TEXT].lower() for t in self.tuples] if lower else \
            [t[self.TEXT] for t in self.tuples]

    def untokenize(self):
        return ''.join([t[self.TEXT_WS] for t in self.tuples]).strip()

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.tuples]

    def slice(self, i, j):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)  # Shallow copy: new obj but shared refs
        new_tokens.tuples = self.tuples[i: j]
        return new_tokens


    def ngrams(self, n=1, lower=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.
        Args:
            n: upper limit of ngram length
            lower: lowercases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(lower)
        ngrams = [(s, e + 1) for s in range(len(words))
                  for e in range(s, min(s + n, len(words)))
                  if not _skip(words[s:e + 1])]

        if as_strings:
            ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

        return ngrams



class SimpleTokenizer:
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'  # Letter, number, combining mark
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE | regex.UNICODE | regex.MULTILINE
        )

    def tokenize(self, text):
        tuples = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            token = matches[i].group()  # Text
            s, t = matches[i].span()
            if i + 1 < len(matches):
                t = matches[i + 1].span()[0]
            tuples.append((token, text[s: t], (s, t)))
        return Tokens(tuples)

########## END: Adapted from https://github.com/facebookresearch/DrQA ##########


def has_answer(candidate, answers, mode='string', tokenizer=None, lower=True):
    if mode == 'string':  # True if candidate['text'] contains an answer.
        assert tokenizer is not None
        text_words = tokenizer.tokenize(uninorm(candidate['text'])).words(
            lower=lower)
        for answer in answers:
            answer_words = tokenizer.tokenize(uninorm(answer)).words(
                lower=lower)
            if contains(text_words, answer_words):
                return True
        return False

    elif mode == 'title':  # True if candidate['title'] matches an answer.
        title = candidate['title'].strip()
        if lower:
            title = title.lower()
        for answer in answers:
            answer = answer.strip()
            if lower:
                answer = answer.lower()
            if title == answer:
                return True
        return False

    else:
        raise ValueError('Invalid has_answer mode: ' + mode)


def uninorm(text): # https://en.wikipedia.org/wiki/Unicode_equivalence
    return unicodedata.normalize('NFD', text)


def omit_ends(string, omit_char):
    if len(string) < 2:
        return string
    return string[1:-1] if string[0] == omit_char and string[-1] == omit_char \
        else string


def topk_retrieval_accuracy(results, k_values, unnormalized=False):
    # Each result has candidates. Each candidate has 'has_answer'.
    k_num_correct = Counter()
    for k in k_values:
        k_num_correct[k] = 0
    for result in results:
        rank_min = float('inf')
        candidates = result['candidates'] if 'candidates' in result else \
                     result['ctxs']  # DPR released
        for rank, candidate in enumerate(candidates):
            if candidate['has_answer']:
                rank_min = rank
                break
        for k in k_values:
            if rank_min < k:
                k_num_correct[k] += 1

    num_queries = len(results)
    def get_acc(num_correct):
        return num_correct if unnormalized else num_correct / num_queries
    k_accuracy = {k: get_acc(num_correct) for k, num_correct in
                  k_num_correct.items()}
    k_accuracy['num_queries'] = num_queries
    return k_accuracy


def print_performance(file_k_num_correct, k_values, micro=False):
    # file_k_num_correct[file(path)] =  {k: num_correct_k (ks), num_queries: N}

    def compute_average_accuracy(k):
        if micro:  # Micro accuracy: makes sense when files same domain/nature
            num_correct_total = sum(file_k_num_correct[f][k] for f in
                                    file_k_num_correct)
            num_queries_total = sum(file_k_num_correct[f]['num_queries'] for f
                                    in file_k_num_correct)
            acc_micro = num_correct_total / num_queries_total
            return acc_micro
        else:  # Macro accuracy: makes more sense when files different domains
            accs = [file_k_num_correct[f][k] / \
                    file_k_num_correct[f]['num_queries'] for f in
                    file_k_num_correct]
            acc_macro = sum(accs) / len(accs)
            return acc_macro

    AVG = 'Avg' + ('(micro)' if micro else '(macro)')
    file_name = {f: os.path.basename(f) for f in file_k_num_correct}
    file_name[AVG] = AVG
    files_sorted = sorted(file_k_num_correct.keys(), key=lambda f: file_name[f])
    if len(files_sorted) > 1:
        files_sorted.append(AVG)

    table_lines = []
    for f in files_sorted:
        table_line = []
        for k in k_values:
            if f == AVG:
                acc = compute_average_accuracy(k)
            else:
                acc = file_k_num_correct[f][k] / \
                      file_k_num_correct[f]['num_queries']
            table_line.append(f'{acc:.1%}')
        table_lines.append(table_line + [file_name[f]] +
                           [str(file_k_num_correct[f]['num_queries'])
                            if f != AVG else '-'])

    header = [f'k={k}' for k in k_values] + ['filename', 'num_queries']
    print('\t'.join(header))
    for line in table_lines:
        print('\t'.join(line))
