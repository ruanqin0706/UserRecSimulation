import numpy as np
import scipy.stats as stats


def get_truncated_norm(num_samples, mu, sigma, lower, upper):
    return stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(num_samples)


def get_truncated_log_norm(mu, sigma, total, upper_bound):
    choose_set = set()
    while len(choose_set) < total:
        val = int(np.random.lognormal(mu, sigma, 1)[0])
        if val <= upper_bound:
            choose_set.add(val)
    return choose_set


def get_truncated_norm_special(num_samples, mu, sigma, lower, upper):
    # return num_samples of integers, non-duplicated
    distribution = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    generator_bool = True
    val_list = list()
    while generator_bool:
        val = int(distribution.rvs(1))
        if val in val_list:
            continue
        else:
            val_list.append(val)

        if len(val_list) == num_samples:
            generator_bool = False
    return np.asarray(val_list)


def get_truncated_norm_pdf(samples, mu, sigma, lower, upper):
    return stats.truncnorm.pdf(samples, (lower - mu) / sigma, (upper - mu) / sigma, mu, sigma)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def ordered_set(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
