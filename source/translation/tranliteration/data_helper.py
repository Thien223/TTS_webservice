import os
import torch
import shutil
from abc import *
import sentencepiece as spm
from torch.utils.data import Dataset


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_BOS = "<s>"
_EOS = "</s>"
_UNK = "<unk>"
_PAD = "<pad>"
_START_VOCAB = [_BOS, _EOS, _UNK, _PAD]

BOS_ID = 0
EOS_ID = 1
UNK_ID = 2
PAD_ID = 3


def basic_tokenizer(sentence):
    return list(sentence.lower().strip())


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size, tokenizer=None):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with open(data_path, encoding='utf-8') as data:
        lines = data.readlines()
        counter = 0
        for i in lines:
            counter += 1
            if counter % 100000 == 0:
                print("  processing line %d" % counter)
            tokens = tokenizer(i) if tokenizer else basic_tokenizer(i)
            for w in tokens:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1

        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]
    f = open(vocabulary_path, 'w', encoding='UTF8')
    for i in vocab_list:
        f.write(i + "\n")
    f.close()


def create_or_get_voca(vocabulary_path):
    with open(vocabulary_path, 'r', encoding='utf-8') as data:
        rev_vocab = [line.strip() for line in data.readlines()]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]

