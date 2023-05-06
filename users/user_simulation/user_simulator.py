from user_simulation.utils import get_truncated_log_norm, get_truncated_norm_special, get_truncated_norm_pdf, \
    ordered_set, softmax
import numpy as np
import random


def simulate_user2(user_seed, user_id,
                   step_no_list,
                   prob_bias_val,
                   info2nid_dict,
                   theme_id_arr, theme_pdf_arr,
                   day_reading_upper_bound, day_reading_freq_mu, day_reading_freq_sigma, ):
    random.seed(user_seed)
    np.random.seed(user_seed)

    reading_list = list()
    for step_no in step_no_list:
        num_readings = get_truncated_log_norm(mu=day_reading_freq_mu,
                                              sigma=day_reading_freq_sigma,
                                              total=1,
                                              upper_bound=day_reading_upper_bound).pop()

        # max mode fetch items mode
        max_size = day_reading_upper_bound * 200

        theme_id_list = np.random.choice(theme_id_arr, max_size, replace=True, p=theme_pdf_arr)

        prob_bias_types = [prob_bias_val, 1 - prob_bias_val]
        is_bias_list = np.random.choice([True, False], max_size, replace=True, p=prob_bias_types)

        nid_list = [np.random.choice(info2nid_dict[(step_no, theme_id, is_bias)], 1, replace=False)[0]
                    for theme_id, is_bias in zip(theme_id_list, is_bias_list)
                    if (step_no, theme_id, is_bias) in info2nid_dict]
        nid_list = ordered_set(nid_list)
        if num_readings > len(nid_list):
            print(f"uid: {user_id} at {step_no} hits condition! "
                  f"the num_readings is {num_readings}, total is {len(nid_list)}")
        reading_list.extend(ordered_set(nid_list)[:num_readings])

    return {
        "uid": user_id,
        "user_seed": user_seed,
        "user_type": 'empty',
        "history_nid_set": reading_list,
        "prob_bias_val": prob_bias_val,
        "theme_id_arr": theme_id_arr,
        "theme_pdf_arr": theme_pdf_arr
    }


def simulate_user(user_seed, user_id,
                  date_info_list, info2nid_dict,
                  theme_id_arr, theme_pdf_arr,
                  user_type,
                  prob_bias_val,
                  day_reading_upper_bound, day_reading_freq_mu, day_reading_freq_sigma, ):
    random.seed(user_seed)
    np.random.seed(user_seed)

    day2prob = {
        "day1": (1 / 7, 6 / 7),
        "day2": (2 / 7, 5 / 7),
        "day3": (3 / 7, 4 / 7),
        "day4": (4 / 7, 3 / 7),
        "day5": (5 / 7, 2 / 7),
        "day6": (6 / 7, 1 / 7),
        "day7": (1, 0)
    }
    reading_list = list()
    for date_info in date_info_list:
        if user_type == 0:
            day_type = np.random.choice(["day1", "day2"], 1, replace=False)[0]
        elif user_type == 1:
            day_type = np.random.choice(["day3", "day4"], 1, replace=False)[0]
        else:
            day_type = np.random.choice(["day5", "day6", "day7"], 1, replace=False)[0]

        is_reading = np.random.choice([True, False], 1, replace=False, p=day2prob[day_type])[0]

        if is_reading:
            num_readings = get_truncated_log_norm(mu=day_reading_freq_mu,
                                                  sigma=day_reading_freq_sigma,
                                                  total=1,
                                                  upper_bound=day_reading_upper_bound).pop()

            # max mode fetch items mode
            max_size = day_reading_upper_bound * 200

            theme_id_list = np.random.choice(theme_id_arr, max_size, replace=True, p=theme_pdf_arr)

            prob_bias_types = [prob_bias_val, 1 - prob_bias_val]
            is_bias_list = np.random.choice([True, False], max_size, replace=True, p=prob_bias_types)

            nid_list = [np.random.choice(info2nid_dict[(date_info, theme_id, is_bias)], 1, replace=False)[0]
                        for theme_id, is_bias in zip(theme_id_list, is_bias_list)
                        if (date_info, theme_id, is_bias) in info2nid_dict]
            nid_list = ordered_set(nid_list)
            if num_readings > len(nid_list):
                print(f"uid: {user_id} at {date_info} hits condition! "
                      f"the num_readings is {num_readings}, total is {len(nid_list)}")
            reading_list.extend(ordered_set(nid_list)[:num_readings])

    return {
        "uid": user_id,
        "user_seed": user_seed,
        "user_type": user_type,
        "history_nid_set": reading_list,
        "prob_bias_val": prob_bias_val,
        "theme_id_arr": theme_id_arr,
        "theme_pdf_arr": theme_pdf_arr
    }


def gen_user_theme_preference(user_seed,
                              theme_size,
                              total_interests,
                              user_interests_sigma,
                              interests_epsilon
                              ):
    random.seed(user_seed)
    np.random.seed(user_seed)

    interests_idx_arr = get_truncated_norm_special(num_samples=theme_size,
                                                   mu=total_interests // 2,
                                                   sigma=user_interests_sigma,
                                                   lower=0,
                                                   upper=total_interests)
    interests_pdf_arr = get_truncated_norm_pdf(samples=interests_idx_arr,
                                               mu=total_interests // 2,
                                               sigma=user_interests_sigma,
                                               lower=0,
                                               upper=total_interests)

    norm_interests_pdf_arr = softmax(interests_pdf_arr)
    norm_interests_pdf_arr = norm_interests_pdf_arr * (1 - interests_epsilon)

    epsilon_idx_arr = [idx for idx in range(total_interests) if idx not in interests_idx_arr]
    epsilon_pdf_arr = np.ones(len(epsilon_idx_arr)) * (interests_epsilon / len(epsilon_idx_arr))

    total_theme_id_arr = np.arange(total_interests)
    np.random.shuffle(total_theme_id_arr)
    theme_id_arr = total_theme_id_arr[np.concatenate((interests_idx_arr, epsilon_idx_arr), axis=0)]  # 获取指定索引的主题编号
    theme_pdf_arr = np.concatenate((norm_interests_pdf_arr, epsilon_pdf_arr), axis=0)
    return theme_id_arr, theme_pdf_arr


def gen_user_single_theme_preference(theme_id_arr, interested_theme_idx, ):
    theme_pdf_arr = np.zeros_like(theme_id_arr, dtype=np.float)
    theme_pdf_arr[interested_theme_idx] = 1.0
    return theme_pdf_arr


def gen_user_multiple_theme_preference(theme_id_arr, interested_theme_idx_list, interests_epsilon):
    total_theme = theme_id_arr.shape[0]
    num_interested_theme = len(interested_theme_idx_list)

    theme_pdf_arr = np.zeros_like(theme_id_arr, dtype=np.float)
    interested_prob_val = (1.0 - interests_epsilon) / num_interested_theme
    other_prob_val = interests_epsilon / (total_theme - num_interested_theme)
    for theme_id in range(total_theme):
        if theme_id in interested_theme_idx_list:
            theme_pdf_arr[theme_id] = interested_prob_val
        else:
            theme_pdf_arr[theme_id] = other_prob_val
    return theme_pdf_arr
