#!/usr/bin/env python

import glob
import itertools
import os

import helper
import model
import scores
from bash import call

#? first of all, clean work and draw folders
for f in glob.glob("w/*") + glob.glob("draw/*"):
    os.remove(f)

#? process files and create lexicon
train_set, train_wrd_tag, test_set, lex = helper.preprocess_files()

#? Create transducers (iob and iob_and_w)

model.create_transducer_from_data([tuple(pair)  # tag - word (condition is tag)
                                   for sentence in train_set
                                   for pair in sentence],
                                  "iob_tagger_trans") # name of our transducer


model.create_transducer_from_data([tuple(pair)  # word - improved iob (condition is improved iob)
                                   for sentence in train_wrd_tag
                                   for pair in sentence],
                                  "iob_and_w_tagger_trans") # name of our transducer


#? Create ngram based language model ---------------------------------------------------------------------

# Here we perform a 'grid search' on all methods and ngram lengths (both for the IOB and IOB+word tagged data)
# so we can see which had the best performance

ngrammake_methods = ["absolute", "katz", "kneser_ney",
                     "presmoothed", "unsmoothed", "witten_bell"]
# "katz_frac", gives error?

ngramcount_orders = ["1", "2", "3", "4", "5"]
version = ["iob", "iob_and_w"]

for kind, method, count in itertools.product(version, ngrammake_methods, ngramcount_orders):

    # if score file for current pair (method, count) has already been calculated and saved to file then skip
    # and continue with next step
    if os.path.exists(f"scores/{kind}_method-{method}_order-{count}.txt"):
        continue

    print(
        f"Processing:\t version: {kind} \t method: {method} \t order: {count}")

    # this is also in charge of concatenating ngram model + transducer

    if kind == "iob":  # create model with smoothing and ngram order
        model.create_iob_ngram_model(method, count)

    elif kind == "iob_and_w":
        model.create_iob_and_wrds_ngram_model(method, count)

    # process all test sentences with the created model
    scores.process_test_sentences(test_set, kind)

    scores.process_preds_to_score(f"{kind}_method-{method}_order-{count}.txt")


# finally, if baseline scores haven't been calculated yet, we calculate them
if not os.path.exists(f"scores/baseline.txt"):
    print("Creating baseline model and processing test set with it")
    model.create_baseline_model()
    scores.process_test_sentences(test_set, "baseline")
    scores.process_preds_to_score("baseline.txt")


# finally run notebook to generate graphics needed for report
call("jupyter nbconvert --execute Graphics.ipynb")