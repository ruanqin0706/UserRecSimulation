import argparse
import os
import pickle
import time

from sklearn.decomposition import LatentDirichletAllocation


def batch_method(data_vectorized, random_state, write_dir):
    lda_model = LatentDirichletAllocation(n_components=n_components,  # Number of topics
                                          learning_method='batch',
                                          random_state=random_state,  # Random state
                                          evaluate_every=-1,  # compute perplexity every n iters, default: Don't
                                          n_jobs=-1,  # Use all available CPUs
                                          )
    lda_output = lda_model.fit_transform(data_vectorized)
    print(lda_model)  # Model attributes

    # Log Likelyhood: Higher the better
    print("Log Likelihood: ", lda_model.score(data_vectorized))

    # Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
    print("Perplexity: ", lda_model.perplexity(data_vectorized))

    # See model parameters
    print(lda_model.get_params())

    with open(os.path.join(write_dir, f"lda_{n_components}.pkl"), "wb") as f:
        pickle.dump(lda_model, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_components", type=int)
    parser.add_argument("--max_components", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--random_state", type=int, default=2022)
    parser.add_argument("--vectorized_path", type=str, )
    parser.add_argument("--write_dir", type=str, )

    args = parser.parse_args()
    os.makedirs(args.write_dir, exist_ok=True)

    with open(args.vectorized_path, "rb") as f:
        data_vectorized = pickle.load(f)
    for idx, n_components in enumerate(range(args.min_components, args.max_components + 1, args.steps)):
        start_time = time.time()
        print(f"start idx: {idx}, n_components: {n_components}")
        batch_method(data_vectorized, args.random_state, args.write_dir)
        end_time = time.time()
        print(f"finish idx: {idx}, costs {(end_time - start_time) / 60} min")
