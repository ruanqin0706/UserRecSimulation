import argparse

import pandas as pd
import os
import pickle
import numpy as np


def generate_pseudo_input(original_behaviors_path,
                          original_news_path,
                          nid2bias_path,
                          write_dir,
                          pseudo_step,
                          news_candidate_strategy,
                          news_candidate_mode,
                          ):
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
                if step < pseudo_step:
                    choice_list.extend(choice_list)
                else:
                    1 / 0
            return " ".join("".join(uid_choice.split("-")) for uid_choice in choice_list) + " " + history_str

    def apply_dynamic_candidate(bias_prop, bias_list, unbias_list):
        assert len(bias_list) == len(unbias_list)
        selected_total = len(bias_list)
        selected_bias = int(selected_total * bias_prop)
        selected_unbias = selected_total - selected_bias
        selected_list = np.random.choice(bias_list, selected_bias, replace=False).tolist()
        selected_unbias_list = np.random.choice(unbias_list, selected_unbias, replace=False).tolist()
        selected_list.extend(selected_unbias_list)
        return selected_list

    assert news_candidate_mode in ["l", "ul"], "The mode of news candidate should be l or ul."

    behaviors_df = pd.read_csv(original_behaviors_path, sep='\t', header=None)
    behaviors_df.columns = ['idx', 'uid', 'history', ]
    behaviors_df.insert(2, "date", "")

    if os.path.isfile(os.path.join(write_dir, "user_choice.pkl")):
        with open(os.path.join(write_dir, "user_choice.pkl"), "rb") as f:
            uid2choice = pickle.load(f)
    else:
        uid2choice = dict()

    behaviors_df["pseudo_history"] = behaviors_df.apply(
        lambda row: gen_pseudo_history(row['uid'], row['history'], uid2choice), axis=1
    )

    behaviors_df.drop(["history", ], axis=1, inplace=True)

    # load original news
    with open(nid2bias_path, "rb") as f:
        nid2bias = pickle.load(f)
    news_df = pd.read_csv(original_news_path, header=None, sep="\t")
    news_df.columns = ['nid', 'category', 'sub_category', 'title', 'abstract', 'url',
                       'title entity', 'abstract entity',
                       'date', 'step', 'bias']
    if news_candidate_strategy == 'fixed':
        candidate_nid_list = news_df[news_df['step'] == pseudo_step]['nid'].tolist()

        candidate2bias_prob = {nid: nid2bias[nid] for nid in candidate_nid_list}
        with open(os.path.join(write_dir, f"candidate2biasprob_pseudo{pseudo_step}.pkl"), "wb") as f:
            pickle.dump(candidate2bias_prob, f)

        candidate_nid_list = ["".join(nid.split("-")) for nid in candidate_nid_list]
        if news_candidate_mode == "l":
            behaviors_df["pseudo_candidate"] = " ".join(f"{n}-1" for n in candidate_nid_list)
        else:
            behaviors_df["pseudo_candidate"] = " ".join(candidate_nid_list)

    elif news_candidate_strategy == 'dynamic':
        candidate_nid_list_bias = news_df[(news_df['step'] == pseudo_step) & (news_df['bias'] == True)]['nid'].tolist()
        candidate_nid_list_unbias = news_df[(news_df['step'] == pseudo_step) & (news_df['bias'] == False)][
            'nid'].tolist()
        behaviors_df["bias_prop"] = [0.1, ] * 250 + [0.2, ] * 250 + [0.3, ] * 250 + [0.4, ] * 250 + [0.5, ] * 250 + [
            0.6, ] * 250 + [0.7, ] * 250 + [0.8, ] * 250
        behaviors_df["pseudo_candidate"] = behaviors_df.apply(
            lambda row: apply_dynamic_candidate(
                bias_prop=row["bias_prop"],
                bias_list=candidate_nid_list_bias,
                unbias_list=candidate_nid_list_unbias), axis=1
        )
        del behaviors_df["bias_prop"]
        candidate_nid_list = behaviors_df["pseudo_candidate"].tolist()
        with open(os.path.join(write_dir, f"candidate_nids_pseudo{pseudo_step}.pkl"), "wb") as f:
            pickle.dump(candidate_nid_list, f)
        candidate2bias_prob = dict()
        candidate_fmt_list = []
        for candidate_nids in candidate_nid_list:
            nid_fmt_list = []
            for nid in candidate_nids:
                if nid not in candidate2bias_prob:
                    candidate2bias_prob[nid] = nid2bias[nid]

                nid_fmt = "".join(nid.split("-"))
                if news_candidate_mode == "l":
                    nid_fmt = f"{nid_fmt}-1"
                nid_fmt_list.append(nid_fmt)
            nid_fmt_str = " ".join(nid_fmt_list)
            candidate_fmt_list.append(nid_fmt_str)
        behaviors_df["pseudo_candidate"] = candidate_fmt_list
        with open(os.path.join(write_dir, f"candidate2biasprob_pseudo{pseudo_step}.pkl"), "wb") as f:
            pickle.dump(candidate2bias_prob, f)
    else:
        1 / 0

    behaviors_file_path = os.path.join(write_dir, f"behaviors_pseudo{pseudo_step}.tsv")
    behaviors_df.to_csv(behaviors_file_path, index=None, header=None, sep="\t")

    # generate current and before news based on pseudo_step
    news_info_df = news_df[news_df['step'] <= pseudo_step]
    news_info_df.insert(0, 'fmt_nid', ["".join(nid.split("-")) for nid in news_info_df['nid'].tolist()])
    del news_info_df['nid'], news_info_df['date'], news_info_df['step'], news_info_df['bias'], news_df
    news_file_path = os.path.join(write_dir, f"news_pseudo{pseudo_step}.tsv")
    news_info_df.to_csv(news_file_path, index=None, header=None, sep="\t")

    print(f"{pseudo_step}:behaviors simulation: "
          f"the shape of behaviors: {behaviors_df.shape}, the shape of news: {news_info_df.shape}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_behaviors_path", type=str, default="")
    parser.add_argument("--original_news_path", type=str, default="")
    parser.add_argument("--nid2bias_path", type=str, default="")
    parser.add_argument("--write_dir", type=str, default="")
    parser.add_argument("--pseudo_step", type=int, default=0)
    parser.add_argument("--candidate_news_mode", type=str, choices=["l", "ul"], default="l")
    parser.add_argument("--news_candidate_strategy", type=str, choices=["fixed", "dynamic"], default="dynamic")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    generate_pseudo_input(original_behaviors_path=args.original_behaviors_path,
                          original_news_path=args.original_news_path,
                          nid2bias_path=args.nid2bias_path,
                          write_dir=args.write_dir,
                          pseudo_step=args.pseudo_step,
                          news_candidate_strategy=args.news_candidate_strategy,
                          news_candidate_mode=args.candidate_news_mode,
                          )
