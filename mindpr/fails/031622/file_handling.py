import csv
import glob
import json
import random

from pathlib import Path


##### BEGIN: Adapted from https://github.com/princeton-nlp/EntityQuestions #####

def read_file(infile, handle_file, verbose=False, skip_first_line=False):
    if verbose:
        print(f'Opening "{infile}"...')
    data = None
    with open(infile) as f:
        if skip_first_line:
            f.readline()
        data = handle_file(f)
    if verbose:
        print('  Done.')
    return data


def read_json(infile, verbose=False):
    handler = lambda f: json.load(f)
    return read_file(infile, handler, verbose=verbose)


def read_jsonl(infile, verbose=False):
    handler = lambda f: [json.loads(line) for line in f.readlines()]
    return read_file(infile, handler, verbose=verbose)


def read_tsv(infile, row_fn=lambda x: x, verbose=False, skip_first_line=False):
    handler = lambda f: [row_fn(line.split('\t')) for line in f.readlines()]
    return read_file(infile, handler, verbose=verbose,
                     skip_first_line=skip_first_line)


def read_csv(infile, delimiter='\t', verbose=False, skip_first_line=False):
    handler = lambda f: list(csv.reader(f, delimiter=delimiter))
    return read_file(infile, handler, verbose=verbose,
                     skip_first_line=skip_first_line)


def write_file(outfile, handle_file, verbose=False):
    if verbose:
        print(f'Writing to "{outfile}"...')
    with open(outfile, 'w+') as f:
        handle_file(f)
    if verbose:
        print('  Done.')


def write_json(outfile, data, verbose=False, pretty=False):
    handler = lambda f: f.write(json.dumps(data, indent=4 if pretty else None))
    write_file(outfile, handler, verbose=verbose)


def write_jsonl(outfile, dictionaries, verbose=False):
    def _write_jsonl(f):
        for dictionary in dictionaries:
            f.write(json.dumps(dictionary) + '\n')
    write_file(outfile, _write_jsonl, verbose=verbose)


def mkdir_optional(outdir):  # Create parents as needed, ignore exists error
    Path(outdir).mkdir(parents=True, exist_ok=True)


def empty_dir(outdir, pattern=None):
    path_object = Path(outdir)
    pattern = '*' if pattern is None else pattern
    for child in Path(outdir).glob(pattern):
        if child.is_file():
            child.unlink()


###### END: Adapted from https://github.com/princeton-nlp/EntityQuestions ######
