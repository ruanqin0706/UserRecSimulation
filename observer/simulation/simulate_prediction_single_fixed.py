import argparse
import os
import heapq
import pickle
import numpy as np


def generate_prediction_res(top_n,
                            pseudo_step,
                            write_dir):
    with open(os.path.join(write_dir, f"candidate2biasprob_pseudo{pseudo_step}.pkl"), "rb") as f:
        info_dict = pickle.load(f)
    candidate_nid = list(info_dict.keys())

    with open(os.path.join(write_dir, f"behaviors_pseudo{pseudo_step}_predictions.pkl"), "rb") as f:
        res_list = pickle.load(f)

    top_pred_arr = []
    top_bias_arr = []
    top_nid_arr = []
    for res in res_list:
        aa = np.asarray(res)
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--pseudo_step", type=str, default="2017-06-26")
    parser.add_argument("--write_dir", type=str, default="")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_prediction_res(top_n=args.top_k,
                            pseudo_step=args.pseudo_step,
                            write_dir=args.write_dir)
