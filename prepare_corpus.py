import argparse

import pandas as pd

DATA_COLS = ['user_id', 'item_id', 'rating', 'timestamp']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--source_data', type=str, default='./data/ratings.dat', help="source data of user-item rankings")
    parser.add_argument('--full_corpus_path', type=str, default='./data/corpus.txt', help="path to save corpus")
    parser.add_argument('--train_corpus_path', type=str, default='./data/train_corpus.txt', help="path to save train corpus")
    parser.add_argument('--valid_path', type=str, default='./data/valid.txt', help="path to save validation")
    parser.add_argument('--test_path', type=str, default='./data/test.txt', help="path to save test")
    parser.add_argument('--pos_thresh', type=float, default=3.5, help="rank threshold to assign for positive items")
    parser.add_argument('--min_items', type=int, default=2, help="min number of positive items needed to store a user")
    return parser.parse_args()


def filter_group(group, pos_thresh, min_items):
    ret_group = group[group['rating'] >= pos_thresh]
    if ret_group.empty or ret_group.shape[0] < min_items:
        print(f'user {group.name} ranked less than {min_items} items above {pos_thresh}')
        return []
    else:
        return ret_group['item_id'].sort_values(by='timestamp').tolist()


def split_train_valid_test(lsts, corpus_path, train_corpus_path, valid_path, test_path):
    with open(train_corpus_path, 'a') as corpus_train_file, open(corpus_path, 'a') as corpus_full_file, \
            open(valid_path, 'a') as valid_file, open(test_path, 'a') as test_file:
        valid_file.write('user_id,item_id\n')
        for u in range(lsts.shape[0]):
            u_lst = lsts[u]
            if len(u_lst):
                target_test_item = u_lst[-1]
                test_file.write(str(u) + ',' + str(target_test_item) + '\n')
                u_lst.remove(target_test_item)
                target_valid_item = u_lst[-1]
                valid_file.write(str(u) + ',' + str(target_valid_item) + '\n')
                corpus_full_file.write(' '.join([str(i) for i in u_lst]) + '\n')
                u_lst.remove(target_valid_item)
                corpus_train_file.write(' '.join([str(i) for i in u_lst]) + '\n')
            else:
                corpus_full_file.write('' + '\n')
                corpus_train_file.write('' + '\n')


def read_data(path, data_cols):
    data = pd.read_csv(path, delimiter='::', names=data_cols, engine='python')
    data[['user_id', 'item_id']] = data[['user_id', 'item_id']].apply(lambda col: col-1)
    return data


def main():
    args = parse_args()
    data = read_data(args.source_data, DATA_COLS)
    users2items = data.groupby('user_id').apply(lambda group: filter_group(group, args.pos_thresh, args.min_items))
    print(f'number of users: {len([user for user in users2items if len(user)])}, number of items: '
          f'{len(set([items for item in users2items.tolist() for items in item]))}')
    split_train_valid_test(users2items, args.full_corpus_path, args.train_corpus_path, args.valid_path, args.test_path)


if __name__ == '__main__':
    main()






