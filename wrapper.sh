#!/bin/bash

# If to check what data base I want to run.
if $1 == "movielens"; then
  python prepare_corpus.py

  python preprocess.py \
--vocab ./data/full_train.txt \
--full_corpus ./data/full_train.txt \
--test_corpus ./data/test.txt \
--build_train_valid \
--train_corpus ./data/train.txt \
--valid_corpus ./data/valid.txt \
--full_train_file ./data/full_train.dat \
--train_file ./data/train.dat \
--valid_file ./data/valid_.dat \
--test_file ./data/test.dat
fi

if $1 == "netflix"; then
  python prepare_corpus.py \
  --input_file ./data/netflix_corpus.csv \
  --line_sep , \
  --min_usr_len 3 \
  --max_usr_len 2700 \
  --min_items_cnt 100 \
  --max_items_cnt 130000 \
  --final_usr_len 3

python preprocess.py \
--vocab ./data/full_train.txt \
--full_corpus ./data/full_train.txt \
--test_corpus ./data/test.txt \
--build_train_valid \
--train_corpus ./data/train.txt \
--valid_corpus ./data/valid.txt \
--full_train_file ./data/full_train.dat \
--train_file ./data/train.dat \
--valid_file ./data/valid_.dat \
--test_file ./data/test.dat
fi

list=[,,,]

# A for loop to run on all the databases.
for database in $list; do
python hyper_param_tune.py \
--model ai2v \
--data_dir ./data/ \
--save_dir ./output/ \
--train train.dat \
--valid valid.dat \
--test test.dat \
--full_train full_train.dat \
--max_epoch 50 \
--patience 3 \
--trials 5 \
--cuda \
--log_dir my_log_dir \
--k 20 \
--hr_out ./output/hr_out.csv \
--rr_out ./output/rr_out.csv \
--cnfg_out ./output/best_cnfg.pkl
done



python evaluation.py --k 20 --model ai2v --test ./data/test.dat