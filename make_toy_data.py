# python make_toy_data.py downloads/data/retriever/nq-train.json downloads/data/retriever/nq-train10.json --N 10
import argparse
import json

from copy import deepcopy


def main(args):
    print(args)
    with open(args.data) as f:
        examples = json.load(f)
    toy_data = []
    for example in examples[:args.N]:
        toy = deepcopy(example)
        toy['positive_ctxs'] = toy['positive_ctxs'][:args.N]
        toy['negative_ctxs'] = toy['negative_ctxs'][:args.N]
        toy['hard_negative_ctxs'] = toy['hard_negative_ctxs'][:args.N]
        toy_data.append(toy)

    with open(args.outfile, 'w+') as f:
        f.write(json.dumps(toy_data, indent=4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, help='DPR retriever training data')
    parser.add_argument('outfile', type=str)
    parser.add_argument('--N', type=int, default=10, help='max num of things')
    args = parser.parse_args()

    main(args)
