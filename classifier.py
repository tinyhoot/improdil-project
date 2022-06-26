#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# classifier.py

from pathlib import Path
from random import Random
from typing import Iterable
import glob
import math

from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


class HateSpeechClassifier:

    def __init__(self, files: list[Path], metadata_csv: Path, eval_files: list[Path] = None, split_corpus: bool = True,
                 ngram_size: tuple[int, int] = (3, 3)):
        """A classifier intended to detect hate speech in a sentence.

        :param files: The corpus to train (and possibly evaluate) on.
        :param metadata_csv: A comma-separated csv containing filenames in column 0 and a hate/noHate label in column 4.
        :param eval_files: If passed, use this list of files for evaluation.
        :param split_corpus: If True and no eval_files were given, split the corpus into train-eval sets automatically.
        :param ngram_size: A tuple with the minimum and maximum sizes (inclusive) of ngrams to consider.
        """
        self.corpus = self.__expand_globs(files)
        # Set up the train-eval split, if necessary.
        if eval_files:
            self._corpus_eval = self.__expand_globs(eval_files)
            self._corpus_train = self.corpus
        elif split_corpus:
            self.__split_train_eval()
        else:
            self._corpus_eval = None
            self._corpus_train = self.corpus

        self.meta_csv = metadata_csv
        self.ngram_size = ngram_size

        # Set up instance attributes for fully trained models.
        self._naive_bayes: MultinomialNB = None
        self._vocab: dict = None

    def __get_ngrams(self, content: Iterable, input_type: str, vocab=None) -> (csr_matrix, CountVectorizer):
        """Extract all n-grams of the given size from the corpus and transform them into machine-usable format.

        I.e., extract the features (X) from the data.

        :param content: An Iterable containing filenames, files or strings to be processed.
        :param input_type: Determines how $content is handled. Must be one of [ filename | file | content ]
        :param ngram_size: The minimum and maximum sizes of the ngrams to be extracted.
        """
        vec = CountVectorizer(input=input_type, ngram_range=self.ngram_size, vocabulary=vocab, analyzer="char_wb",
                              max_df=0.8)
        # If the vocabulary is already defined, we just need a transformation that exists in relation to something
        # we previously transformed, i.e. to training data.
        if vocab:
            doc_term_freqs = vec.transform(content)
        else:
            doc_term_freqs = vec.fit_transform(content)
        return doc_term_freqs, vec

    def __get_target_values(self, feature_files: list[Path]) -> list[int]:
        """Get a list containing the target value y for each corresponding file.

        I.e., extract the expected response (y) from the data.
        """
        # The memory overhead in this method is pretty nuts (3*len(corpus)). I haven't managed to come up with a more
        # elegant solution, but the risk of all this exceeding the available memory is very low, since it's just about
        # the files' names rather than their contents.
        meta_labels = {}
        corpus_filenames = [path.stem for path in feature_files]
        with open(self.meta_csv, "r", encoding="utf-8") as csv:
            # Skip the header.
            next(csv)
            for line in csv:
                split = line.split(",")
                filename = split[0]
                if filename in corpus_filenames:
                    # Files labelled as hate are considered a 1.
                    label = 1 if split[4].lower().startswith("hate") else 0
                    meta_labels[filename] = label

        if len(feature_files) > len(meta_labels):
            raise ValueError(f"The annotations csv does not contain enough information to cover all files in the "
                             f"corpus! ({len(meta_labels)} < {len(feature_files)})")
        # Ensure the list with the target values is sorted in the same order as the training corpus.
        # Thanks to the logic above, the dict lookups are guaranteed to succeed.
        return [meta_labels[path.stem] for path in feature_files]

    def __expand_globs(self, files: list[Path]) -> list[Path]:
        """Replace all globs in the corpus ("filenames" with an asterisk in them) with the proper filenames.

        This likely won't come into play when properly deployed and executed from the terminal, but PyCharm doesn't
        do it on its own :(
        """
        # Create a new list with all the proper filenames to replace the globby one.
        new_corpus = []
        for path in files:
            if "*" in str(path):
                # Found a glob, resolve it.
                new_corpus += [Path(res_path) for res_path in glob.glob(str(path))]
            else:
                # Continue using the original path if it was "normal".
                new_corpus.append(path)

        return new_corpus

    def __split_train_eval(self, eval_ratio: float = 0.2):
        """Split the corpus into training and evaluation sets.

        By default, uses 20% of the available data for evaluation.
        """
        eval_size = math.floor(len(self.corpus) * eval_ratio)

        # Choose a portion of the corpus at random.
        rand = Random()
        self._corpus_eval = rand.sample(self.corpus, eval_size)

        # Remove evaluation files from the training corpus.
        # Sets are fine here since the ordering does not matter *yet*, and we do not want duplicates anyway.
        self._corpus_train = list(set(self.corpus) - set(self._corpus_eval))

    def evaluate(self, eval_files: list[Path] = None) -> tuple[float, float, float, None]:
        """Evaluate the performance of a model.

        Requires a trained model to be loaded.

        :return: A tuple containing precision, recall, f-score, and support.
        """
        if not self._naive_bayes or not self._vocab:
            raise RuntimeError("No trained model is currently loaded!")
        if not eval_files:
            if not self._corpus_eval:
                raise ValueError("Cannot evaluate on an empty list of files!")
            eval_files = self._corpus_eval

        response_obs = self.__get_target_values(eval_files)
        response_pred = self.predict(eval_files, "filename")

        return metrics.precision_recall_fscore_support(response_obs, response_pred, average="binary")

    def predict(self, observations: Iterable, input_type: str) -> list[int]:
        """Predict whether a sequence of sentences or files contains hate speech.

        Requires a trained model to be loaded.

        :param observations: An iterable containing sentences or files to predict on.
        :param input_type: Determines how observations are handled. Must be one of [ filename | file | content ]
        :return: For each sentence, 1 if it contains hate speech and 0 otherwise.
        """
        if not self._naive_bayes or not self._vocab:
            raise RuntimeError("No trained model is currently loaded!")

        features, _ = self.__get_ngrams(observations, input_type=input_type, vocab=self._vocab)
        return self._naive_bayes.predict(features)

    def train(self) -> MultinomialNB:
        """Train the classifier on the provided corpus."""
        # Extract the training values.
        features, vec = self.__get_ngrams(self._corpus_train, "filename")
        target_y = self.__get_target_values(self._corpus_train)

        # Train the model.
        naive_bayes = MultinomialNB()
        naive_bayes.fit(features, target_y)
        self._naive_bayes = naive_bayes
        self._vocab = vec.vocabulary_
        return naive_bayes

    def serialise(self, filename: str = "hsmodel.joblib"):
        """Save a trained model to disk."""
        joblib.dump(self, filename)

    @staticmethod
    def load(filename: str = "hsmodel.joblib") -> "HateSpeechClassifier":
        """Load a trained model from disk."""
        return joblib.load(filename)
