# IMPRODIL Semester Project

Classify sentences based on hate speech.

## Introduction

With the meteoric rise of online platforms, automated forms of hate speech detection are becoming increasingly relevant.
In their paper, Schmidt & Wiegand (2017) claim that n-grams are highly predictive for hate speech, although their
claim is mostly aimed at comment-level detection (e.g. an entire tweet). Out of curiosity, this program thus aims to
detect hate speech at the sentence level and see how useful n-grams are in this instance.

To that end, a simple Multinomial Naive Bayes classifier has been trained on an annotated corpus of a white supremacist
forum assembled by de Gibert et al. (2018). This program lets you predict whether a sentence contains hate speech using
the pretrained model, or train and/or evaluate based on your own data.

## Prerequisites

* Python 3.9+
* scikit-learn 1.0.2
* A full list of dependencies can be found in `requirements.txt`

## Usage

The program can be broadly divided into three parts: general usage, training, and evaluating.

For simply having it predict on a sentence, use the `-s` flag. The program then returns the sentence index or corpus
filename, the prediction, and the original sentence. Note that a 1 signals hate speech in the sentence, whereas a 0
means that none was detected.

```sh
$ python3 main.py -s "A mean sentence full of hate" -s "A sentence full of love"
s0  1   A mean sentence full of hate
s1  0   A sentence full of love
```

The program can also take entire corpora at a time, assuming that each file contains exactly one sentence.

```sh
$ python3 main.py -c data/all_files/*.txt
138407.txt  0   It was a nice day outside
...
```

The output from either of these commands can also be rerouted into a specified file using the `-o` flag.

By default, the program assumes that a `pretrained_model.joblib` is present in the current directory. A different model
for predicting can be specified using `-m`.

```sh
$ python3 main.py -c data/all_files/*.txt -o output.txt -m hsmodel.joblib
...
```

### Training

If you want to train your own model, use the `train` subcommand. This command requires two positional arguments: `csv`
which denotes the filepath to a .csv file containing annotations of the training corpus, and `corpus`, which takes an
arbitrary number of filepaths to the training corpus. See the Data section for more information on corpus structure.

The program will then train on the corpus and save the completed model to disk. You can also change the output
file by passing the `-o` flag.

```sh
$ python3 main.py train data/annotations.csv data/sampled_train/*.txt
Trained model saved to hsmodel.joblib
```

If you wish to have the program immediately evaluate the newly trained model, pass the `--eval` flag. The program will
then choose a random sample of 20% of the training corpus to later evaluate against. See the Evaluation section below.

```sh
$ python3 main.py train data/annotations.csv data/all_files/*.txt --eval
```

### Evaluation

A trained model can be evaluated at any time using the `eval` subcommand. Three positional arguments are required:
* The path to the model to evaluate
* The path to the annotations csv for the evaluation corpus
* The path or paths to the evaluation corpus itself.

```sh
$ python3 main.py eval pretrained_model.joblib data/annotations.csv data/sampled_test/*.txt
Precision:      0.77109440267335
Recall:         0.9644723092998955
F-Score:        0.8570102135561745
```

## Data

The `data` directory contains the annotated English hate speech corpus assembled by de Gibert et al. The corpus consists of two
parts. 

Three folders containing text files. While `all_files` contains the entire corpus, the two sampled folders were assembled
by the corpus authors and each contain a fraction of the dataset.
Each sentence in the corpus has been assigned its own file, following a pattern denoting the id of the original
forum post followed by the number of the sentence in the post: `31783672_1.txt` `31783672_3.txt`

In addition, an `annotations.csv` contains metadata for each file in the corpus. The relevant parts for this project are
the filename in the first column, and hate speech annotation in the fifth.

The corpus is available online at [https://github.com/Vicomtech/hate-speech-dataset](https://github.com/Vicomtech/hate-speech-dataset)

## References

O. de Gibert, N. Perez, A. García-Pablos, M. Cuadros. (2018). Hate Speech Dataset from a
White Supremacy Forum. Proceedings of the 2nd Workshop on Abusive Language Online, 11-20.

Schmidt, A., & Wiegand, M. (2017). A Survey on Hate Speech Detection using Natural Language Processing. Proceedings of
the Fifth International Workshop on Natural Language Processing for Social Media, 1–10. 
[https://doi.org/10.18653/v1/W17-1101](https://doi.org/10.18653/v1/W17-1101)