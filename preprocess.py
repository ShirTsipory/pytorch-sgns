# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
import pathlib


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/full_corpus.txt', help="corpus path for building vocab")
    parser.add_argument('--full_corpus', type=str, default='./data/full_corpus.txt', help="corpus path")
    parser.add_argument('--test_corpus', type=str, default='./data/test_corpus.txt', help="test corpus path")
    parser.add_argument('--build_train_valid', action='store_true', help="build part train and validation sets from provided paths")
    parser.add_argument('--train_corpus', type=str, default='./data/train_corpus.txt', help="part train corpus path to build part train set, in case build_train_valid is True")
    parser.add_argument('--valid_corpus', type=str, default='./data/valid_corpus.txt', help="validation corpys path to build validation set, in case build_train_valid is True")
    parser.add_argument('--full_train_file', type=str, default='./data/full_train.dat', help="full train file name")
    parser.add_argument('--train_file', type=str, default='./data/train.dat', help="train file name, in case of build_valid")
    parser.add_argument('--valid_file', type=str, default='./data/valid.dat', help="validation file name, in case of build_valid")
    parser.add_argument('--test_file', type=str, default='./data/test.dat', help="test file of sub users and target items")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    parser.add_argument('--max_user', type=int,default=1000, help='maximum length of usr')
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, window=5, unk='<UNK>', pad='pad', data_dir='./data/'):
        self.window = window
        self.unk = unk
        self.data_dir = data_dir
        self.pad = pad
        self.wc = {}

    def build(self, filepath, max_vocab=20000):
        print("building vocab...")
        step = 0
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                user = line.split()
                for item in user:
                    self.wc[item] = self.wc.get(item, 0) + 1
        print("")
        # sorted list of items in a descent order of their frequency
        self.wc[self.unk] = 1
        self.wc[self.pad] = 1
        self.idx2item = sorted(self.wc, key=self.wc.get, reverse=True)[:max_vocab - 1]
        self.item2idx = {self.idx2item[idx]: idx for idx, _ in enumerate(self.idx2item)}
        self.vocab = set([item for item in self.item2idx])
        pickle.dump(self.wc, open(os.path.join(self.data_dir, 'ic.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.data_dir, 'vocab.dat'), 'wb'))
        pickle.dump(self.idx2item, open(os.path.join(self.data_dir, 'idx2item.dat'), 'wb'))
        pickle.dump(self.item2idx, open(os.path.join(self.data_dir, 'item2idx.dat'), 'wb'))
        print("build done")

    def create_train_samp(self, user, item_target, pad_idx, max_user):
        sub_user = user[:item_target]
        # pad_times = max_user - len(sub_user)
        # sub_user += pad_times * ['pad']
        target_item = user[item_target]
        return [self.item2idx[item] for item in sub_user], self.item2idx[target_item]

    def convert(self, filepath, savepath, max_user):
        print("converting corpus...")
        step = 0
        data = []
        usrs_len = []
        item2idx = pickle.load(pathlib.Path(self.data_dir, 'item2idx.dat').open('rb'))
        pad_idx = item2idx['pad']
        with codecs.open(filepath, 'r', encoding='utf-8') as file:
            num_users = 0
            for line in file:
                step += 1
                if not step % 1000:
                    print("working on {}kth line".format(step // 1000), end='\r')
                line = line.strip()
                if not line:
                    continue
                user = []
                for item in line.split():
                    if item in self.vocab:
                        user.append(item)
                    else:
                        user.append(self.unk)
                usrs_len.append(len(user))
                if len(user) < 2:
                    print('skip user')
                    continue
                num_users += 1
                # split large users to sub users
                sub_users = [user[x:x+max_user] for x in range(0, len(user), max_user)]
                for user in sub_users:
                    for item_target in range(1, len(user)):
                        data.append((self.create_train_samp(user, item_target, pad_idx, max_user)))

        print("")
        pickle.dump(data, open(savepath, 'wb'))
        print("conversion done")
        print("num of users:", num_users)
        print("max user:", max(usrs_len))


def main():
    args = parse_args()
    preprocess = Preprocess(unk=args.unk, data_dir=args.data_dir)
    preprocess.build(args.vocab, max_vocab=args.max_vocab)
    preprocess.convert(args.full_corpus, args.full_train_file, args.max_user)
    preprocess.convert(args.test_corpus, args.test_file, args.max_user)
    if args.build_train_valid:
        preprocess.convert(args.train_corpus, args.train_file, args.max_user)
        preprocess.convert(args.valid_corpus, args.valid_file, args.max_user)


if __name__ == '__main__':
    main()
