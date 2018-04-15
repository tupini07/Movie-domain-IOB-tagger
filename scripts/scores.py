import os
import re

import pandas
import seaborn as sns

from bash import call
from tqdm import tqdm

sns.set_context("paper", font_scale=1.2)


def process_preds_to_score(score_file_name):
    """
    passes the `w/pred_coneval.txt` file through the conlleval script and saves the resulting scores in 
     `scores/<score_file_name>` file
    """
    call(
        f"cat w/pred_coneval.txt | ../P1_data/scripts/conlleval.pl > scores/{score_file_name}")


def process_test_sentences(sentences, kind):
    """
    Processes all test sentences with the model of type `kind` and saves the predictions
    to a file
    """

    def process_test_sentence(sent):
        """ 
        Helper function that processes a test sentence indivivdually
        """
        
        # create a transducer for the test sentence
        # it transduces from word -> word
        with open("w/test_sent_fsa.txt", "w") as test_file:
            i = 0

            for w, _ in sent:  # we only write the word and keep the test label for testing
                test_file.write(f"{i}\t{i+1}\t{w}\t{w}\n")
                i += 1

            test_file.write(f"{i}")

        # complite the fst for the test sentence and compute the predictions with the model we're testing. 
        # finally save the predictions to a file
        call(f"fstcompile --isymbols=w/lex.syms --osymbols=w/lex.syms --keep_osymbols --keep_isymbols w/test_sent_fsa.txt | " +
             f"fstcompose - w/{kind}_wfst_ngrm.fsa | fstrmepsilon | fstshortestpath | fsttopsort | " +
             f"fstprint - " +
             f" > w/prediction_on_sent.txt")

        # read the file and extract predictions (x, y)
        pd = pandas.read_csv(
            "w/prediction_on_sent.txt", delimiter="\t", header=None)

        pd = pd[:-1][[2, 3]].get_values() # obtain only columns for word and tag (discard: id and weight columns)
        pd = [[w, re.sub(r"__.*", "", t)] for w, t in pd] # replace the O__word tags with only O

        return pd # return word and predicted tag

    ff = open("w/pred_coneval.txt", "w")

    for sent in tqdm(sentences): # tqdm provides nice progress bar

        preds = process_test_sentence(sent) # get dataframe representation of predictions

        # appende predictions for specific sentence to the pred_coneval.txt file
        # what we write is: word, correct-tag, predicted-tag
        # which is the format accepted by the conlleval script. We separate the predictions for 
        # each sentence with an empty line (the \n\n)
        ff.write("\n".join(
            w + " " + sent[i][1] + " " + p for i, (w, p) in enumerate(preds)) + "\n\n")


def process_score_files():
    """
    Finds scores in scores/ folder, parses them and processes them (generete graph). 
    returns the method, order that gave the best "f1" 
    """

    def extract_metrics_from_file(filename):
        # second line of file holds metrics
        cnts = open("scores/" + filename, "r").read().split("\n")[1]
        cnts = re.sub(":|%", "", cnts).split("; ")
        cnts = {m: float(v) for
                m, v in [l.split("  ") for l in cnts]}

        return cnts


    csv = "version,method,order,FB1,accuracy,precision,recall\n"

    for ff in os.listdir("scores/"):

        if ff == "baseline.txt": # if the file we're processing is the baseline scores file
            version = "Baseline"
            method = "None"
            order = "1"

        else:
            # extract information from filename
            match = re.search(r"(iob|iob_and_w)_method-(.*)_order-(.*).txt", ff)

            version = match.group(1) # iob or iob_and_w (basic or improved)
            method = match.group(2) # smoothing method
            order = match.group(3) # n-gram length
            
        metrics = extract_metrics_from_file(ff)

        # add information to a CSV file (so we can process it later on)
        csv += f"{version},{method},{order},{metrics['FB1']},{metrics['accuracy']},{metrics['precision']},{metrics['recall']}\n"

    # save results in a CSV in case we want to process them later
    # with something else
    with open("w/csv_rep_scores.csv", "w") as csvr:
        csvr.write(csv)

    # and we also get a free pandas dataframe
    df = pandas.read_csv("w/csv_rep_scores.csv")

    return df
