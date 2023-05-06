import argparse
import os
import pickle
from collections import OrderedDict
import numpy as np
import pandas as pd


def generate_pseudo_feedback(write_dir,
                             pseudo_step,
                             original_behaviors_path,
                             user_bias_prop_list,
                             selected_n,
                             strategy):
    with open(os.path.join(write_dir, f"top_nid_{pseudo_step}.pkl"), "rb") as f:
        top_nid_arr = pickle.load(f)
    with open(os.path.join(write_dir, f"top_bias_{pseudo_step}.pkl"), "rb") as f:
        top_bias_arr = pickle.load(f)

    if os.path.isfile(os.path.join(write_dir, "user_choice.pkl")):
        with open(os.path.join(write_dir, "user_choice.pkl"), "rb") as f:
            choice_dict = pickle.load(f)
    else:
        choice_dict = dict()

    uid_list = pd.read_csv(original_behaviors_path, header=None, sep="\t")[1].tolist()
    for rdx, (uid, top_nid, top_bias) in enumerate(zip(uid_list, top_nid_arr, top_bias_arr)):

        uid_seed = int(uid[3:]) + int(pseudo_step)
        np.random.seed(uid_seed)

        num_items = top_nid.shape[0]
        if strategy == 1:
            items_idx_arr = np.arange(num_items)
            choice_idx_arr = np.random.choice(items_idx_arr, selected_n, replace=False)
            uid_choice_arr = top_nid[choice_idx_arr]
            choice_bias_arr = top_bias[choice_idx_arr]
        elif strategy == 2:
            uid_choice_arr = top_nid[:selected_n]
            choice_bias_arr = top_bias[:selected_n]
        elif strategy == 3:
            num_rank = np.arange(num_items) + 1
            prob_rank = num_rank / np.sum(num_rank)
            log2_rank = np.log2(prob_rank) / np.sum(np.log2(prob_rank))
            items_idx_arr = np.arange(num_items)
            choice_idx_arr = np.random.choice(items_idx_arr, selected_n, p=log2_rank, replace=False)
            uid_choice_arr = top_nid[choice_idx_arr]
            choice_bias_arr = top_bias[choice_idx_arr]
        elif strategy == 4:
            user_bias_preference = user_bias_prop_list[rdx]
            top_user_preference = []
            for bias_val in top_bias:
                if bias_val >= 0.5:
                    top_user_preference.append(user_bias_preference)
                else:
                    top_user_preference.append(1 - user_bias_preference)
            top_user_preference = np.asarray(top_user_preference)
            preference_rank = top_user_preference / np.sum(top_user_preference)

            items_idx_arr = np.arange(num_items)
            choice_idx_arr = np.random.choice(items_idx_arr, selected_n, p=preference_rank, replace=False)
            uid_choice_arr = top_nid[choice_idx_arr]
            choice_bias_arr = top_bias[choice_idx_arr]
        elif strategy == 5:
            user_bias_preference = user_bias_prop_list[rdx]
            top_user_preference = []
            for bias_val in top_bias:
                if bias_val < 0.5:
                    top_user_preference.append(user_bias_preference)
                else:
                    top_user_preference.append(1 - user_bias_preference)
            top_user_preference = np.asarray(top_user_preference)
            preference_rank = top_user_preference / np.sum(top_user_preference)

            items_idx_arr = np.arange(num_items)
            choice_idx_arr = np.random.choice(items_idx_arr, selected_n, p=preference_rank, replace=False)
            uid_choice_arr = top_nid[choice_idx_arr]
            choice_bias_arr = top_bias[choice_idx_arr]
        elif strategy == 6:
            user_bias_preference = user_bias_prop_list[rdx]
            num_rank = np.arange(num_items) + 1
            prob_rank = num_rank / np.sum(num_rank)
            log2_rank = np.log2(prob_rank) / np.sum(np.log2(prob_rank))

            top_user_preference = []
            for bias_val in top_bias:
                if bias_val >= 0.5:
                    top_user_preference.append(user_bias_preference)
                else:
                    top_user_preference.append(1 - user_bias_preference)
            top_user_preference = np.asarray(top_user_preference)
            preference_rank = top_user_preference / np.sum(top_user_preference)

            avg_rank = (log2_rank + preference_rank) / 2

            items_idx_arr = np.arange(num_items)
            choice_idx_arr = np.random.choice(items_idx_arr, selected_n, p=avg_rank, replace=False)
            uid_choice_arr = top_nid[choice_idx_arr]
            choice_bias_arr = top_bias[choice_idx_arr]
        elif strategy == 7:
            user_bias_preference = user_bias_prop_list[rdx]
            num_rank = np.arange(num_items) + 1
            prob_rank = num_rank / np.sum(num_rank)
            log2_rank = np.log2(prob_rank) / np.sum(np.log2(prob_rank))

            top_user_preference = []
            for bias_val in top_bias:
                if bias_val < 0.5:
                    top_user_preference.append(user_bias_preference)
                else:
                    top_user_preference.append(1 - user_bias_preference)
            top_user_preference = np.asarray(top_user_preference)
            preference_rank = top_user_preference / np.sum(top_user_preference)

            avg_rank = (log2_rank + preference_rank) / 2

            items_idx_arr = np.arange(num_items)
            choice_idx_arr = np.random.choice(items_idx_arr, selected_n, p=avg_rank, replace=False)
            uid_choice_arr = top_nid[choice_idx_arr]
            choice_bias_arr = top_bias[choice_idx_arr]
        else:
            1 / 0

        if uid not in choice_dict:
            choice_dict[uid] = dict()
            choice_dict[uid]["choice"] = OrderedDict()
            choice_dict[uid]["bias"] = OrderedDict()

        choice_dict[uid]["choice"][pseudo_step] = uid_choice_arr.tolist()
        choice_dict[uid]["bias"][pseudo_step] = choice_bias_arr.tolist()

    with open(os.path.join(write_dir, "user_choice.pkl"), "wb") as f:
        pickle.dump(choice_dict, f)

    print(f"{pseudo_step}:behaviors feedback finished!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nid2bias_path", type=str, default="")
    parser.add_argument("--original_behaviors_path", type=str, )
    parser.add_argument("--pseudo_step", type=int, )
    parser.add_argument("--write_dir", type=str)
    parser.add_argument("--selected_n", type=int)
    parser.add_argument("--strategy", type=int)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.strategy in (4, 5, 6, 7):

        def gen_pseudo_history(uid, history, uid2choice):

            history_str = " ".join(["".join(nid.split("-")) for nid in history.split(" ")])

            if uid not in uid2choice:
                return history_str
            else:
                choice_list = []
                step2choice = uid2choice[uid]["choice"]
                step_reversed = reversed(list(step2choice.keys()))
                for step in step_reversed:
                    choice_list = step2choice[step]
                    if step < args.pseudo_step:
                        choice_list.extend(choice_list)
                    else:
                        1 / 0
                return " ".join("".join(uid_choice.split("-")) for uid_choice in choice_list) + " " + history_str


        def apply_dynamic_candidate(history, nid2bias, ):
            num_bias = 0
            num_unbias = 0
            for nid in history.split(" "):
                if nid2bias[f"{nid[:2]}-{nid[2:7]}-{nid[7:]}"] >= 0.5:
                    num_bias += 1
                else:
                    num_unbias += 1
            bias_prop = num_bias / (num_bias + num_unbias)
            return bias_prop


        if os.path.isfile(os.path.join(args.write_dir, "user_choice.pkl")):
            with open(os.path.join(args.write_dir, "user_choice.pkl"), "rb") as f:
                uid2choice = pickle.load(f)
        else:
            uid2choice = dict()
        # load original news
        with open(args.nid2bias_path, "rb") as f:
            nid2bias = pickle.load(f)

        behaviors_df = pd.read_csv(args.original_behaviors_path, sep='\t', header=None)
        behaviors_df.columns = ['idx', 'uid', 'history', ]
        behaviors_df["pseudo_history"] = behaviors_df.apply(
            lambda row: gen_pseudo_history(row['uid'], row['history'], uid2choice), axis=1
        )
        behaviors_df["user_bias_prop"] = behaviors_df.apply(
            lambda row: apply_dynamic_candidate(history=row["pseudo_history"],
                                                nid2bias=nid2bias, ), axis=1
        )
        user_bias_prop_list = behaviors_df["user_bias_prop"].tolist()
    else:
        user_bias_prop_list = []
    print(args.strategy, type(args.strategy))
    generate_pseudo_feedback(write_dir=args.write_dir,
                             pseudo_step=args.pseudo_step,
                             original_behaviors_path=args.original_behaviors_path,
                             user_bias_prop_list=user_bias_prop_list,
                             selected_n=args.selected_n,
                             strategy=args.strategy)
