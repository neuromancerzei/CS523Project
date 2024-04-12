"""
generate word2vec model, and save to word2vec.model
generate market-aux-vectors_data.csv
"""
import collections
from typing import Callable
import datasets
import gensim.models.word2vec
import pandas
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import tqdm
import spacy
import nltk
import os
import re
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
from langdetect import detect
from imblearn.over_sampling import SMOTE
from keras.preprocessing.sequence import pad_sequences
from multiprocessing import Pool
from gensim.models import Word2Vec
import numpy
from gensim.models.phrases import Phrases
# from gensim.summarization import summarize # conda install -c conda-forge gensim
import pyarrow
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import LineSentence

DEBUG = False
# set the random seeds for torch and numpy. This is to ensure this program is reproducable, i.e. we get the same results each time we run it.
seed = 1234

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


class Tokenization(object):
    NLTK = 'nltk'
    SPACY = 'spacy'
    BASIC = 'basic'

    nltk.download('stopwords')
    nltk.download('punkt')

    @staticmethod
    def get_nltk():
        """https://www.nltk.org/"""
        return nltk.word_tokenize

    @staticmethod
    def get_spacy():
        """
        https://spacy.io/
        python -m spacy download en_core_web_sm
        """
        nlp = spacy.load("en_core_web_sm")
        return nlp

    @staticmethod
    def get_basic():
        tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
        return tokenizer

    @staticmethod
    def chain(text: str) -> list[str]:
        # Clean the Text: Remove any irrelevant characters (like punctuation, numbers, or special characters) that don’t contribute to sentiment or meaning.
        # Stop Words Removal: Filter out common words (like “the”, “is”, “in”, etc.) that appear frequently across all text but carry little meaning.
        text = Tokenization.convert_to_lowercase(
            Tokenization.remove_special_characters(Tokenization.remove_html_tags(text)))
        # Stemming/Lemmatization: Reduce words to their root form. Lemmatization considers the context and converts the word to its meaningful base form, whereas stemming simply removes suffixes.
        # Tokenization: Break the text down into individual words or tokens. This is a fundamental step in text processing.
        tokens = Tokenization.stem_text(Tokenization.remove_stopwords(Tokenization.tokenize_text(text)))
        return tokens

    @staticmethod
    def remove_html_tags(text):
        """
        Removing HTML Tags
        """
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text

    @staticmethod
    def remove_special_characters(text):
        """
         Removing Special Characters
        :param text:
        :return:
        """
        clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
        return clean_text

    @staticmethod
    def tokenize_text(text):
        """
        Tokenization
        :param text:
        :return:
        """
        tokens = word_tokenize(text)
        return tokens

    @staticmethod
    def convert_to_lowercase(text):
        """
        Lowercasing
        :param text:
        :return:
        """
        lowercased_text = text.lower()
        return lowercased_text

    @staticmethod
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

    @staticmethod
    def stem_text(tokens):
        """
        Stemming and lemmatization are techniques to reduce words to their root forms, which can help group similar words. Stemming is more aggressive and may result in non-dictionary words, whereas lemmatization produces valid words.
        :param tokens:
        :return:
        """
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in tokens]
        return stemmed_tokens

    @staticmethod
    def lemmatize_text(tokens):
        """
        Stemming and lemmatization are techniques to reduce words to their root forms, which can help group similar words. Stemming is more aggressive and may result in non-dictionary words, whereas lemmatization produces valid words.
        :param tokens:
        :return:
        """
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return lemmatized_tokens

    @staticmethod
    def remove_duplicates(texts):
        unique_texts = list(set(texts))
        return unique_texts

    @staticmethod
    def correct_spelling(text):
        """
        Dealing with Noisy Text
        Noisy text data can include typos, abbreviations, non-standard language usage, and other irregularities. Addressing such noise is crucial for ensuring the accuracy of text analysis. Techniques like spell-checking, correction, and custom rules for specific noise patterns can be applied.
        :param text:
        :return:
        """

        spell = SpellChecker()
        tokens = word_tokenize(text)
        corrected_tokens = [spell.correction(word) for word in tokens]
        corrected_text = ' '.join(corrected_tokens)
        return corrected_text

    @staticmethod
    def clean_custom_patterns(text):
        # Example: Replace email addresses with a placeholder
        clean_text = re.sub(r'\S+@\S+', '[email]', text)
        return clean_text

    @staticmethod
    def fix_encoding(text):
        try:
            decoded_text = text.encode('utf-8').decode('utf-8')
        except UnicodeDecodeError:
            decoded_text = 'Encoding Error'
        return decoded_text

    @staticmethod
    def remove_whitespace(text):
        cleaned_text = ' '.join(text.split())
        return cleaned_text

    @staticmethod
    def detect_language(text):
        try:
            language = detect(text)
        except:
            language = 'unknown'
        return language

    @staticmethod
    def balance_text_data(X, y):
        smote = SMOTE(sampling_strategy='auto')
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled

    @staticmethod
    def pad_text_sequences(text_sequences, max_length):
        padded_sequences = pad_sequences(text_sequences, maxlen=max_length, padding='post', truncating='post')
        return padded_sequences

    @staticmethod
    def parallel_process_text(data, cleaning_function, num_workers):
        with Pool(num_workers) as pool:
            cleaned_data = pool.map(cleaning_function, data)
        return cleaned_data

    @staticmethod
    def clean_multilingual_text(text, language_code):
        nlp = spacy.load(language_code)
        doc = nlp(text)
        cleaned_text = ' '.join([token.lemma_ for token in doc])
        return cleaned_text

    # @staticmethod
    # def summarize_long_document(text, ratio=0.2):
    #     summary = summarize(text, ratio=ratio)
    #     return summary


class MarketAuxDataSet(object):
    TEXT_FEATURE = "text"
    SENTIMENT_FEATURE = "sentiment"
    LABEL_FEATURE = "label"

    TOKEN_FEATURE = "tokens"
    ID_FEATURE = "ids"
    VECTOR_FREATURE = "vector"

    NEGATIVE_LABEL = 0
    NEUTRAL_LABEL = 1
    POSITIVE_LABEL = 2

    def __init__(self, df: pandas.DataFrame):
        df = df.reset_index(drop=True)
        self.df = df
        self.index = 0

    def sentiment_to_labels(self):
        self.df[self.LABEL_FEATURE] = self.df[self.SENTIMENT_FEATURE].apply(lambda x: round(x + 1))

    def __getitem__(self, index):
        return self.df.iloc[index]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.df):
            result = self.df.iloc[self.index]
            self.index += 1
            return result
        else:
            self.index = 0
            raise StopIteration

    def split(self, test_size, random_state):
        df1, df2 = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return MarketAuxDataSet(df1), MarketAuxDataSet(df2)

    def map(self, col: str, rst_col: str, fn):
        self.df[rst_col] = self.df[col].apply(fn)
        return self

    def __len__(self):
        return len(self.df)


class SentimentAnalysisData(object):
    vector_size = 300
    WORD2VEC_MODEL_PATH = f"dataset/word2vec{vector_size}.model"
    UNK_KEY = "<unk>"
    PAD_KEY = "<pad>"

    def __init__(self):
        self.data: MarketAuxDataSet
        self.train_data: MarketAuxDataSet
        self.valid_data: MarketAuxDataSet
        self.test_data: MarketAuxDataSet
        self.vocab: torchtext.vocab.Vocab
        self.model: gensim.models.word2vec.Word2Vec

    def __call__(self, *args, **kwargs):
        self.load_dataset()
        self.tokenization()
        self.create_vocabulary()
        self.numericalizing_data()
        self.build_word2vec()
        self.save_data()

    @property
    def num_class(self):
        return self.train_data.features['label'].numclasses

    @property
    def labels_names(self):
        return self.train_data.features['label'].names

    def load_dataset(self):
        self._load_market_ananlysis()
        # if DEBUG:
        #     self.train_data_debug, self.valid_data_debug, self.test_data_debug = self._load_imbd()

    def _load_market_ananlysis(self):
        df1 = pandas.read_csv("dataset/market-aux.csv")
        df2 = pandas.read_csv("dataset/market-aux/2024-03-01.csv")
        df = pandas.concat([df1, df2], axis=0)
        self.data = MarketAuxDataSet(df)
        self.data.sentiment_to_labels()

    def _split_data(self):
        train_data, test_data = self.data.split(test_size=0.5, random_state=42)
        train_data, valid_data = train_data.split(test_size=0.5, random_state=42)
        return train_data, valid_data, test_data

    def _load_imbd(self):
        train_data, test_data = datasets.load_dataset("imdb", split=["train", "test"])
        train_valid_data = train_data.train_test_split(test_size=0.25)
        train_data, valid_data = train_valid_data["train"], train_valid_data["test"]
        return train_data, valid_data, test_data

    @classmethod
    def arrow_token_fn(cls, item: dict, tokenizer: Callable[[str], list] = Tokenization.chain):
        tokens = tokenizer(item["text"])
        return {MarketAuxDataSet.TOKEN_FEATURE: tokens}

    @classmethod
    def df_token_fn(cls, tokenizer: Callable[[str], list] = Tokenization.chain):
        def inner(text):
            return tokenizer(text)

        return inner

    def tokenization(self):
        map_args = (MarketAuxDataSet.TEXT_FEATURE, MarketAuxDataSet.TOKEN_FEATURE, self.df_token_fn())
        self.data = self.data.map(*map_args)
        # self.train_data = self.test_data.map(*map_args)
        # self.valid_data = self.valid_data.map(*map_args)
        # self.test_data = self.test_data.map(*map_args)
        #
        # if DEBUG:
        #     self.train_data_debug = self.train_data_debug.map(self.arrow_token_fn)
        #     self.valid_data_debug = self.valid_data_debug.map(self.arrow_token_fn)
        #     self.test_data_debug = self.test_data_debug.map(self.arrow_token_fn)

    def create_vocabulary(self):
        def build_vocab(data):
            min_freq = 1
            UNK = "<unk>"
            PAD = "<pad>"

            vocab = torchtext.vocab.build_vocab_from_iterator(self.data.df[MarketAuxDataSet.TOKEN_FEATURE],
                                                              min_freq=min_freq,
                                                              specials=[UNK, PAD])
            unk_index = vocab[UNK]
            pad_index = vocab[PAD]
            vocab.set_default_index(unk_index)
            return vocab

        self.vocab = build_vocab(self.data)
        # if DEBUG:
        #     self.vocab_debug = build_vocab(self.train_data_debug)

    @classmethod
    def arrow_numericalizing_fn(cls, item: dict, vocab: torchtext.vocab.Vocab):
        ids = vocab.lookup_indices(item[MarketAuxDataSet.TOKEN_FEATURE])
        return {MarketAuxDataSet.ID_FEATURE: ids}

    @classmethod
    def df_numericalizing_fn(cls, vocab: torchtext.vocab.Vocab):
        def inner(text):
            return vocab.lookup_indices(text)

        return inner

    def numericalizing_data(self):
        map_args = (MarketAuxDataSet.TOKEN_FEATURE, MarketAuxDataSet.ID_FEATURE, self.df_numericalizing_fn(self.vocab))
        self.data.map(*map_args)
        # self.train_data.map(*map_args)
        # self.valid_data.map(*map_args)
        # self.test_data.map(*map_args)
        # if DEBUG:
        #     self.train_data_debug = self.train_data_debug.map(self.arrow_numericalizing_fn,
        #                                                       fn_kwargs={"vocab": self.vocab_debug})
        #     self.valid_data_debug = self.valid_data_debug.map(self.arrow_numericalizing_fn,
        #                                                       fn_kwargs={"vocab": self.vocab_debug})
        #     self.test_data_debug = self.test_data_debug.map(self.arrow_numericalizing_fn,
        #                                                     fn_kwargs={"vocab": self.vocab_debug})

    @classmethod
    def df_vector_fn(cls, model: gensim.models.word2vec.Word2Vec):
        def inner(tokens):
            return numpy.mean([model.wv[token] for token in tokens], axis=0)

        return inner

    def build_word2vec(self):
        """
        Use models like Word2Vec or GloVe to represent words in a dense vector space where semantically similar words are closer to each other.
        word2vec
        1) Continuous Bag of Words
        2) skip gram
        See https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
        """
        if not os.path.exists(self.WORD2VEC_MODEL_PATH):
            self.model = Word2Vec(sentences=self.data.df["tokens"], vector_size=self.vector_size, window=5, min_count=1)
            # index_to_key = self.model.wv.index_to_key
            # vectors = self.model.wv.vectors
            # v = self.model.wv['sp']
            # key = self.model.wv.key_to_index['stock']
            # top10 = self.model.wv.similar_by_word("stock")
            self.model.wv.add_vector(self.UNK_KEY, numpy.zeros(self.vector_size))
            self.model.wv.add_vector(self.PAD_KEY, numpy.zeros(self.vector_size))
            self.model.save(self.WORD2VEC_MODEL_PATH)

            # phrases = Phrases(self.train_data.df["text"], min_count=1, threshold=1)  # phrase
        else:
            self.model = Word2Vec.load(self.WORD2VEC_MODEL_PATH)
        self.data.df['tokens'].apply(
            lambda tokens: numpy.mean([self.model.wv[token] for token in tokens], axis=0))
        self.data.map(MarketAuxDataSet.TOKEN_FEATURE, MarketAuxDataSet.VECTOR_FREATURE,
                            self.df_vector_fn(self.model))

        # if DEBUG:
        #     self.mode_debug = Word2Vec(sentences=self.train_data_debug.df["tokens"], size=100, window=5, min_count=1)

    def save_data(self):
        selected_colums = [MarketAuxDataSet.VECTOR_FREATURE, MarketAuxDataSet.LABEL_FEATURE]
        self.data.df[selected_colums].to_csv(f"dataset/market-aux-vectors{self.vector_size}_data.csv", index=False)


if __name__ == '__main__':
    sa = SentimentAnalysisData()
    sa()

    print("prepared model data successfully")
