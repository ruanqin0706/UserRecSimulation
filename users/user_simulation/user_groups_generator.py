import argparse
import pickle
import time

from user_simulation.user_simulator import simulate_user, gen_user_theme_preference
from user_simulation.utils import get_truncated_norm

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_prefix", type=str)
    parser.add_argument("--total_users", type=int)
    parser.add_argument("--expected_users", type=int)
    parser.add_argument("--start_idx", type=int)

    parser.add_argument("--write_path", type=str, )

    parser.add_argument("--theme_size_mu", type=float, default=5)
    parser.add_argument("--theme_size_sigma", type=float, default=2.5)
    parser.add_argument("--theme_size_lower", type=int, default=5)
    parser.add_argument("--theme_size_upper", type=int, default=16)

    parser.add_argument("--total_interests", type=int, default=20)
    parser.add_argument("--user_interests_sigma", type=float, default=2.5)
    parser.add_argument("--interests_epsilon", type=float, default=0.1)

    parser.add_argument("--day_reading_freq_mu", type=float, default=1.5)
    parser.add_argument("--day_reading_freq_sigma", type=float, default=1.0)
    parser.add_argument("--day_reading_upper_bound", type=int, default=100)

    parser.add_argument('--prob_user_types', nargs='+', default=[0.5, 0.3, 0.2])

    parser.add_argument("--info2nid_path", type=str, default="")
    parser.add_argument("--date_str_list_path", type=str, default="")
    parser.add_argument("--nid2bias_prob_path", type=str, default="")
    parser.add_argument("--prob_bias_val", type=float)

    return parser.parse_args()


def func():
    args = parse_args()
    with open(args.date_str_list_path, "rb") as f:
        date_str_list = pickle.load(f)
    with open(args.info2nid_path, "rb") as f:
        info2nid = pickle.load(f)
    with open(args.nid2bias_prob_path, "rb") as f:
        nid2bias_prob = pickle.load(f)

    np.random.seed(0)
    user_types = np.random.choice([0, 1, 2], args.expected_users, replace=True, p=args.prob_user_types)
    theme_size_list = [int(val) for val in
                       get_truncated_norm(num_samples=args.expected_users,
                                          mu=args.theme_size_mu, sigma=args.theme_size_sigma,
                                          lower=args.theme_size_lower, upper=args.theme_size_upper)]

    user_list = []
    for idx in range(args.start_idx, args.start_idx + args.total_users):
        print(f"{idx}")
        start_time = time.time()
        user_seed = idx + 1
        user_type = user_types[idx]
        np.random.seed(user_seed)
        theme_id_arr, theme_pdf_arr = gen_user_theme_preference(
            user_seed=user_seed,
            theme_size=theme_size_list[idx],
            total_interests=args.total_interests,
            user_interests_sigma=args.user_interests_sigma,
            interests_epsilon=args.interests_epsilon,
        )
        user_info_dict = simulate_user(
            user_seed=user_seed, user_id=f"{args.user_prefix}{user_seed}",
            date_info_list=date_str_list, info2nid_dict=info2nid,
            theme_id_arr=theme_id_arr, theme_pdf_arr=theme_pdf_arr,
            user_type=user_type,
            prob_bias_val=args.prob_bias_val,
            day_reading_upper_bound=args.day_reading_upper_bound,
            day_reading_freq_mu=args.day_reading_freq_mu,
            day_reading_freq_sigma=args.day_reading_freq_sigma, )
        # calculate user bias based on records in history
        num_bias = 0
        num_unbias = 0
        for nid in user_info_dict["history_nid_set"][::-1]:
            if nid2bias_prob[nid] >= 0.5:
                num_bias += 1
            else:
                num_unbias += 1
        user_bias = num_bias / (num_bias + num_unbias)
        user_info_dict["user_bias"] = user_bias
        user_list.append(user_info_dict)
        end_time = time.time()
        print(f"it costs: {(end_time - start_time) / 60} min.")

    with open(args.write_path, "wb") as f:
        pickle.dump(user_list, f)


if __name__ == '__main__':
    func()
