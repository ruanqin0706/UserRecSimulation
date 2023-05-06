import argparse
import heapq
import os

import pickle
import numpy as np


def generate_prediction_res(
        infer_user_path,
        infer_news_path,
        news_path,
        top_n,
        pseudo_step,
        write_dir):
    # users: idx2vec
    with open(infer_user_path, "rb") as f:
        user_idx2vec = pickle.load(f)
    v_list = []
    for k, v in user_idx2vec.items():
        v_list.append(v)
    user_vec_arr = np.asarray(v_list)

    with open(infer_news_path, "rb") as f:
        all_idx2vec = pickle.load(f)
    all_nid_list = []
    with open(news_path, 'r') as f:
        for line in f:
            nid = line.replace("\n", "").split("\t")[0]
            nid = f"{nid[:2]}-{nid[2:7]}-{nid[7:]}"
            all_nid_list.append(nid)
    all_nid2vec = dict()
    for idx, nid in enumerate(all_nid_list):
        all_nid2vec[nid] = all_idx2vec[idx + 1]

    top_pred_arr = []
    top_bias_arr = []
    top_nid_arr = []

    with open(os.path.join(write_dir, f"candidate2biasprob_pseudo{pseudo_step}.pkl"), "rb") as f:
        info_dict = pickle.load(f)
    with open(os.path.join(write_dir, f"candidate_nids_pseudo{pseudo_step}.pkl"), "rb") as f:
        candidate_nid_list = pickle.load(f)
    assert len(candidate_nid_list) == user_vec_arr.shape[0]

    for zdx, candidate_nid in enumerate(candidate_nid_list):
        candidate_nid_vec_arr = []
        for nid in candidate_nid:
            candidate_nid_vec_arr.append(all_nid2vec[nid])
        candidate_nid_vec_arr = np.asarray(candidate_nid_vec_arr)

        aa = np.dot(user_vec_arr[zdx], candidate_nid_vec_arr.T)
        ind = heapq.nlargest(top_n, range(len(aa)), aa.take)
        top_pred_arr.append([aa[idx] for idx in ind])
        top_nid_arr.append([candidate_nid[idx] for idx in ind])
        top_bias_arr.append([info_dict[candidate_nid[idx]] for idx in ind])

    top_bias_arr = np.asarray(top_bias_arr)
    top_pred_arr = np.asarray(top_pred_arr)
    top_nid_arr = np.asarray(top_nid_arr)
    with open(os.path.join(write_dir, f"top_bias_{pseudo_step}.pkl"), "wb") as f:
        pickle.dump(top_bias_arr, f)

    with open(os.path.join(write_dir, f"top_pred_{pseudo_step}.pkl"), "wb") as f:
        pickle.dump(top_pred_arr, f)

    with open(os.path.join(write_dir, f"top_nid_{pseudo_step}.pkl"), "wb") as f:
        pickle.dump(top_nid_arr, f)

    print(f"{pseudo_step}:behaviors prediction finished.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--news_file", type=str, default="")
    parser.add_argument("--infer_news_file", type=str, default="")
    parser.add_argument("--infer_user_file", type=str, default="")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--pseudo_step", type=str, default="2017-06-26")
    parser.add_argument("--write_dir", type=str, default="")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_prediction_res(
        infer_user_path=args.infer_user_file,
        infer_news_path=args.infer_news_file,
        news_path=args.news_file,
        top_n=args.top_k,
        pseudo_step=args.pseudo_step,
        write_dir=args.write_dir)
