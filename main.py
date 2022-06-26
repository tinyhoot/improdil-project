#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# main.py

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

from classifier import HateSpeechClassifier


def __get_cli() -> ArgumentParser:
    """Get the command line interface for this program."""
    parser = ArgumentParser(description="Hate Speech Classifier")
    subparsers = parser.add_subparsers(help="train or evaluate a model. see train/eval --help")

    # Add the subparser for the "train" command.
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("csv", type=Path, help="the path to the metadata csv")
    train_parser.add_argument("corpus", nargs="+", type=Path, help="the path to the corpus to train on")
    train_parser.add_argument("-o", "--output", type=str, default="hsmodel.joblib",
                              help="where to save the trained model")
    train_parser.add_argument("--eval", action="store_true",
                              help="automatically split the corpus and evaluate the newly trained model")
    train_parser.set_defaults(func=__parse_train)

    # Add the subparser for the "eval" command.
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("model", type=Path, help="the path to a previously trained model")
    eval_parser.add_argument("csv", type=Path, help="the path to the metadata csv")
    eval_parser.add_argument("corpus", nargs="+", type=Path, help="the path to the corpus to evaluate on")
    eval_parser.set_defaults(func=__parse_eval)

    # Define the parser for general usage.
    parser.add_argument("-s", "--sentence", action="append", type=str,
                        help="a sentence to predict hate speech for, can be passed multiple times")
    parser.add_argument("-c", "--corpus", nargs="+", type=Path,
                        help="the path(s) to a corpus containing one sentence per file")
    parser.add_argument("-m", "--model", type=Path, default="pretrained_model.joblib",
                        help="the path to a previously trained model")
    parser.add_argument("-o", "--output", type=str,
                        help="if passed, save output to this file")
    parser.set_defaults(func=__parse_default)

    return parser


def __print_evaluation(evaluation):
    """Neatly display the final stats for the model after evaluation."""
    print(f"Precision:\t{evaluation[0]}")
    print(f"Recall:\t\t{evaluation[1]}")
    print(f"F-Score:\t{evaluation[2]}")


def __print_results_files(files: Iterable[Path], predictions: list, save_file: str = None):
    """Output the results to the command line, for each input file individually."""
    # If desired, output to a file instead.
    if save_file:
        out_file = open(save_file, "w", encoding="utf-8")

    for path, pred in zip(files, predictions):
        with open(path, "r", encoding="utf-8") as file:
            # In this corpus, each file contains only a single line.
            sentence = file.readline()
        out = f"{path.stem}\t{pred}\t{sentence}"
        if save_file:
            out_file.write(out + "\n")
        else:
            print(out)

    if save_file:
        out_file.close()


def __print_results_strings(strings: Iterable[str], predictions: list, save_file: str = None):
    """Output the results to the command line, for each sentence individually."""
    # If desired, output to a file instead.
    if save_file:
        file = open(save_file, "w", encoding="utf-8")

    idx = 0
    for sent, pred in zip(strings, predictions):
        out = f"s{idx}\t{pred}\t{sent}"
        if save_file:
            file.write(out + "\n")
        else:
            print(out)
        idx += 1

    if save_file:
        file.close()


def __parse_eval(args):
    """Process the subcommand for evaluating a model."""
    classifier = HateSpeechClassifier.load(args.model)
    classifier.meta_csv = args.csv
    __print_evaluation(classifier.evaluate(args.corpus))


def __parse_default(args):
    """When no subcommands are given, use this to assume a general use case and predict hate in input."""
    # If no input was given, raise an error.
    if not args.sentence and not args.corpus:
        print("Error: At least one sentence or file must be given to predict on.\n"
              "Example usage: main.py -s 'A very mean and hateful sentence.'")
        return

    classifier = HateSpeechClassifier.load(args.model)
    if args.sentence:
        # Process sentence-based input.
        predictions = classifier.predict(args.sentence, "content")
        __print_results_strings(args.sentence, predictions, args.output)
    if args.corpus:
        # Process file-based input.
        predictions = classifier.predict(args.corpus, "filename")
        __print_results_files(args.corpus, predictions, args.output)


def __parse_train(args):
    """Process the subcommand for training a model."""
    classifier = HateSpeechClassifier(args.corpus, args.csv, ngram_size=(3, 3), split_corpus=args.eval)
    classifier.train()
    classifier.serialise(args.output)
    print(f"Trained model saved to {args.output}")
    # Evaluate the newly trained model.
    if args.eval:
        __print_evaluation(classifier.evaluate())


def main():
    # Get command line arguments.
    parser = __get_cli()
    args = parser.parse_args()

    # Process the arguments based on which subcommand was given (train/eval/none).
    args.func(args)


if __name__ == "__main__":
    main()
