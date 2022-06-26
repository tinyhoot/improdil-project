#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_classifier.py

from pathlib import Path

from classifier import HateSpeechClassifier
import pytest


def test_get_target_values():
    corpus = [Path("test_data/12834278_3.txt"), Path("test_data/12834278_5.txt")]
    target = [0, 1]
    hsc = HateSpeechClassifier(corpus, Path("test_data/annotations.csv"))
    assert target == hsc._HateSpeechClassifier__get_target_values(corpus)


def test_get_target_values_no_metadata():
    corpus = [Path("test_data/12834278_3.txt"), Path("test_data/12834278_5.txt"), Path("test_data/12834278_9.txt")]
    hsc = HateSpeechClassifier(corpus, Path("test_data/annotations.csv"))
    with pytest.raises(ValueError):
        hsc._HateSpeechClassifier__get_target_values(corpus)


def test_expand_globs_no_glob():
    corpus = [Path("test_data/12843037_1.txt"), Path("test_data/12834278_3.txt")]
    target = ["test_data/12843037_1.txt", "test_data/12834278_3.txt"]
    hsc = HateSpeechClassifier(corpus, None)
    attempt = [str(file) for file in hsc._HateSpeechClassifier__expand_globs(corpus)]
    assert target == attempt


def test_expand_globs_glob():
    corpus = [Path("test_data/128342*")]
    target = {"test_data/12834217_2.txt", "test_data/12834217_4.txt", "test_data/12834278_3.txt",
              "test_data/12834278_5.txt", "test_data/12834278_9.txt"}
    hsc = HateSpeechClassifier(corpus, None)
    attempt = {str(file) for file in hsc._HateSpeechClassifier__expand_globs(corpus)}
    assert target == attempt


def test_train_eval_split():
    corpus = [Path("test_data/12834217_2.txt"), Path("test_data/12834217_4.txt"), Path("test_data/12834278_3.txt"),
              Path("test_data/12834278_5.txt"), Path("test_data/12834278_9.txt")]
    hsc = HateSpeechClassifier(corpus, None)
    assert set(corpus) == set(hsc._corpus_train) | set(hsc._corpus_eval)
    assert set() == set(hsc._corpus_train) & set(hsc._corpus_eval)


# Public function testing starts here

@pytest.fixture
def classifier():
    corpus = [Path("test_data/*.txt")]
    meta = Path("test_data/annotations.csv")
    return HateSpeechClassifier(corpus, meta)


def test_evaluate_no_model(classifier):
    with pytest.raises(RuntimeError):
        classifier.evaluate()


def test_evaluate_no_data():
    hsc = HateSpeechClassifier.load("test_data/test.joblib")
    hsc._corpus_eval = None
    with pytest.raises(ValueError):
        hsc.evaluate()


def test_predict_no_model(classifier):
    with pytest.raises(RuntimeError):
        classifier.predict([], "")


