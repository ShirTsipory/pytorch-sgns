import numpy as np
import time
import csv
from collections import Counter
from datetime import datetime
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--usr_pos', type=int, default=0, help="")
    parser.add_argument('--item_pos', type=int, default=1, help="")
    parser.add_argument('--rate_pos', type=int, default=2, help="")
    parser.add_argument('--date_pos', type=int, default=3, help="")
    parser.add_argument('--positive_threshold', type=float, default=4.0, help="")
    parser.add_argument('--input_file', type=str, default='./data/movielens_corpus.dat', help="")
    parser.add_argument('--line_sep', type=str, default='::', help="")
    parser.add_argument('--min_usr_len', type=int, default=2, help="")
    parser.add_argument('--max_usr_len', type=int, default=250, help="")
    parser.add_argument('--min_items_cnt', type=int, default=5, help="")
    parser.add_argument('--max_items_cnt', type=int, default=500, help="")
    parser.add_argument('--final_usr_len', type=int, default=2, help="")
    parser.add_argument('--out_full_train', type=str, default='./data/full_corpus.txt', help="output data path")
    parser.add_argument('--out_test', type=str, default='./data/test_corpus.txt', help="output test data path")
    parser.add_argument('--out_train', type=str, default='./data/train_corpus.txt', help="output train data path")
    parser.add_argument('--out_valid', type=str, default='./data/valid_corpus.txt', help="output validation data path")
    return parser.parse_args()


class User:
    def __init__(self, user_id):
        self.user_id = user_id
        self.items = []

    def ArrangeItemList(self):
        self.items = [item_index[0] for item_index in sorted(self.items, key=lambda x: x[1])]


class Index(object):
    def __init__(self):
        self.item2index = {}
        self.index2item = {}


def IndexLabels(labels, mask_zero=False):
    label2index = {}
    index2label = {}
    for i, label in enumerate(labels):
        if mask_zero:
            i += 1
        label2index[label] = i
        index2label[i] = label
    return label2index, index2label


def CountFilter(counter, min_count=10, max_count=10000000):
    return [item for item, count in counter.most_common(len(counter)) if count > min_count and count < max_count]


def ComputeSplitIndices(num_instances, test_size=0.1):
    permutation = np.random.permutation(num_instances)
    split_index = int((1 - test_size) * num_instances)
    return permutation[:split_index], permutation[split_index:]


def preparation(item_pos, usr_pos, rate_pos, date_pos, positive_threshold, input_file, line_sep, min_usr_len,
                max_usr_len, min_items_cnt, max_items_cnt, final_usr_len, out_full_train, out_test, out_train,
                out_valid):
    user2data = {}
    t = time.clock()
    with open(input_file) as rating_file:
        for i, line in enumerate(rating_file):
            if i == 0:
                continue
            if i % 5000000 == 0:
                print(i)
            line = line.strip().split(line_sep)
            line = [i for i in line if i != '']
            user_id = line[usr_pos]
            if user_id not in user2data:
                user2data[user_id] = User(user_id)
            user = user2data[user_id]
            # treat date format
            try:
                date = int(line[date_pos])

            except:
                date = int(datetime.strptime(line[date_pos], '%Y-%m-%d').timestamp())
            if float(line[rate_pos]) > positive_threshold:
                user.items.append((line[item_pos], date))

    valid_users = []
    for user in list(user2data.values()):
        if len(user.items) > min_usr_len and len(user.items) < max_usr_len:
            user.ArrangeItemList()
            valid_users.append(user.user_id)
    print(len(valid_users))
    print(time.clock() - t)

    np.random.seed(0)

    item_counter = Counter()
    index = Index()

    for user in list(valid_users):
        user = user2data[user]
        item_counter.update(user.items)

    index.item2index, index.index2item = IndexLabels(CountFilter(item_counter, min_count=min_items_cnt, max_count=max_items_cnt), True)

    valid_users_filtered = []
    for user_id in list(valid_users):
        user = user2data[user_id]
        items = [item for item in user.items if item in index.item2index]
        if len(items) > final_usr_len:
            valid_users_filtered.append(user_id)
    valid_users = valid_users_filtered

    train_indices, test_indices = ComputeSplitIndices(len(valid_users), test_size=0.1)
    train_users = [valid_users[i] for i in train_indices]
    train_item_lists = [user2data[user].items for user in train_users]
    test_users = [valid_users[i] for i in test_indices]
    test_item_lists = [user2data[user].items for user in test_users]

    with open(out_full_train, 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(train_item_lists)

    with open(out_test, 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(test_item_lists)

    train_indices, validation_indices = ComputeSplitIndices(len(train_indices), test_size=0.1)
    train_users = [valid_users[i] for i in train_indices]
    train_item_lists = [user2data[user].items for user in train_users]
    validation_users = [valid_users[i] for i in validation_indices]
    validation_item_lists = [user2data[user].items for user in validation_users]

    with open(out_train, 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(train_item_lists)
    with open(out_valid, 'w', newline="") as x:
        csv.writer(x, delimiter=" ").writerows(validation_item_lists)

    print("Items#: ", len(index.item2index))
    print("Full corpus users#:", len(valid_users))
    print("Train users#: ", len(train_users))
    print("validation users#: ", len(validation_users))
    print("Test users#: ", len(test_users))


def main():
    args = parse_args()
    preparation(args.item_pos, args.usr_pos, args.rate_pos, args.date_pos, args.positive_threshold, args.input_file,
                args.line_sep, args.min_usr_len, args.max_usr_len, args.min_items_cnt, args.max_items_cnt,
                args.final_usr_len, args.out_full_train, args.out_test, args.out_train, args.out_valid)


if __name__ == '__main__':
    main()