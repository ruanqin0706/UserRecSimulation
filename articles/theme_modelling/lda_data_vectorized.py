import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

np.random.seed(2022)
article_path = "../dataset/detection/processed/hp_bypublisher_article_filtering.csv"
article_list = []

with open(article_path, "r") as f:
    for article_line in tqdm(f):
        article = article_line.split("***sep***")[-1]
        article_list.append(article)

vectorizer = CountVectorizer(analyzer='word',
                             min_df=10,  # minimum reqd occurences of a word
                             stop_words='english',  # remove stop words
                             lowercase=True,  # convert all words to lowercase
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                             # max_features=50000,             # max number of uniq words
                             )

data_vectorized = vectorizer.fit_transform(article_list)

with open("../dataset/theme_modelling/hp_bypublisher_article_filtering_data_vectorized.pkl", "wb") as f:
    pickle.dump(data_vectorized, f)
