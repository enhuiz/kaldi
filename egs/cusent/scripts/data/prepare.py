#!/usr/bin/env python3
import argparse
import pandas as pd
import string
import unicodedata
import glob
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('corpus', type=Path)
parser.add_argument('--out-dir', type=Path, default='data')
args = parser.parse_args()

# prepare data/train
train = Path(args.out_dir, 'train')
train.mkdir(parents=True, exist_ok=True)


def closure():
    punctuation = '。﹐，、､？！；：“”「」『』（）——-《》……  o'
    punctuation = set(punctuation + string.punctuation)

    def f(s):
        s = [c.lower() for c in s if c not in punctuation]
        s = ' '.join(s)
        return s

    return f


process_orthogra = closure()


def load_orthogra(dir_):
    path = Path(dir_, 'ANNOTATE', 'ORTHOGRA')

    def read_csv(f):
        return pd.read_csv(f, sep='\t', header=None, names=['id', 'orthogra'])

    try:
        with open(path, 'r', encoding='big5hkscs') as f:
            df = read_csv(f)
    except:
        with open(path, 'r', encoding='big5hkscs', errors='ignore') as f:
            df = read_csv(f)
        print('Warning', path, 'has invalid bytes during encoding.')

    df['id'] = dir_.name + '/' + df['id']
    df['orthogra'] = df['orthogra'].apply(process_orthogra)
    return df


def load_phonetic(dir_):
    path = Path(dir_, 'ANNOTATE', 'PHONETIC')
    df = pd.read_csv(path, sep='\t', header=None, names=['id', 'phonetic'])
    df['id'] = dir_.name + '/' + df['id']
    return df


def load_speaker(dir_):
    dir_ = Path(dir_)
    df = pd.merge(load_orthogra(dir_), load_phonetic(dir_), 'left', 'id')

    olen = df['orthogra'].str.split().apply(len)
    plen = df['phonetic'].str.split('-').apply(len)
    df = df[olen == plen]

    return df


wildcard = Path(args.corpus, '**', 'C' + '[A-Z0-9]' * 3 + '[FM]')
dirs = glob.glob(str(wildcard))
df = pd.concat([load_speaker(dir_) for dir_ in dirs])
df = df.reset_index(drop=True)

# prepare data/train/text
with open(Path(train, 'text'), 'w') as f:
    csv = df[['id', 'orthogra']].to_csv(index=None, sep=' ', header=None)
    csv = csv.replace('"', '')
    f.write(csv)

# prepare data/train/words.txt
words = sorted(set([w for s in df['orthogra'].values for w in s.split(' ')]))
with open(Path(train, 'words.txt'), 'w') as f:
    f.write('\n'.join(words) + '\n')

# prepare data/train/lexicon.txt


def closure():
    onset = r'(b|p|m|f|d|t|n|l|g|k|ng|h|gw|kw|w|c|s|j)'
    nuclei = r'(aa|i|u|e|o|yu|oe|eo|a)'
    coda = r'(p|t|k|m|n|ng|i|u)'
    pattern = '{}?{}?{}?\d'.format(onset, nuclei, coda)

    def f(p):
        return ' '.join(re.findall(pattern, p)[0]).strip()

    return f


split_phones = closure()


lexicon = []
for _, row in df.iterrows():
    orthogra = row['orthogra'].split()
    phonetic = row['phonetic'].split('-')
    phonetic = [split_phones(p) for p in phonetic]
    lexicon += list(zip(orthogra, phonetic))

lexicon = sorted(set(lexicon))
with open(Path(train, 'lexicon.txt'), 'w') as f:
    f.write('\n'.join([' '.join(p) for p in lexicon]))
