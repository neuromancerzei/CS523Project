import pandas
from gensim import corpora
from gensim.models import LdaModel
from pprint import pprint
import lda
import lda.datasets
import numpy as np
import collections
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy
from collections import Counter
def remove_special_characters(text):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return clean_text


def remove_html_tags(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text


def stem_text(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return stemmed_tokens


def convert_to_lowercase(text):
    lowercased_text = text.lower()
    return lowercased_text


def remove_stopwords(tokens):
    """
    Stopword Removal
    Stopwords are common words such as “the,” “and,” or “in” that carry little meaningful information in many NLP tasks. Removing stopwords can reduce noise and improve the efficiency of text analysis.
    :param tokens:
    :return:
    """
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens


def tokenize_text(text):
    """
    Tokenization
    :param text:
    :return:
    """
    tokens = word_tokenize(text)
    return tokens


def chain(text):
    text = convert_to_lowercase(
        remove_special_characters(remove_html_tags(text)))
    # Stemming/Lemmatization: Reduce words to their root form. Lemmatization considers the context and converts the word to its meaningful base form, whereas stemming simply removes suffixes.
    # Tokenization: Break the text down into individual words or tokens. This is a fundamental step in text processing.
    tokens = stem_text(remove_stopwords(tokenize_text(text)))
    return tokens


df = pandas.read_csv("dataset/market-aux.csv")
df = df.reset_index(drop=True)
df["tokens"] = df["text"].apply(chain)
word2vec_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=1)
word2vec_model.wv.add_vector("<unk>", numpy.zeros(word2vec_model.vector_size))
unk_idx = word2vec_model.wv.get_index("<unk>")
df["tf"] = df['tokens'].apply(lambda tokens: [Counter(tokens).get(word2vec_model.wv.index_to_key[i], 0) for i in range(word2vec_model.vector_size)])
X2 = numpy.array(df["tf"].tolist())
vocab2= word2vec_model.wv.index_to_key
lda_model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
lda_model.fit(X2)
topic_word = lda_model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab2)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = lda_model.doc_topic_


# 395 2345
# X = lda.datasets.load_reuters()  # 395 4258
# vocab = lda.datasets.load_reuters_vocab()  # 4258
# titles = lda.datasets.load_reuters_titles()  # 395
# model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
# model.fit(X)
# topic_word = model.topic_word_
# n_top_words = 8
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
#     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
#
# doc_topic = model.doc_topic_





if __name__ == '__main__':
    print("Running LDA model...")
