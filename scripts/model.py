import nltk
import math
from bash import call, draw, fstcompile


def create_transducer_from_data(all_training_pairs, name):
    """
    `all_training_pairs` - is an array of pairs, where each pair is [word, tag]
    `name` - is the name we want to give to the transducer

    The transducer is written to a file and then compiled with fstcompile
    """

    # first count occurrences of each word given the tag
    cfd = nltk.ConditionalFreqDist( reversed(pair)  # tag - word (condition is tag)
                                   for pair in all_training_pairs)  # pair [word, tag]

    with open(f"w/{name}.txt", "w") as tagger_trans:

        for word, tag in set(all_training_pairs):  # only unique pairs

            # calculate probability of word given the tag
            # probability(word | tag) =
            #          count(word, tag) / count(tag)

            freqs = cfd[tag]

            val = freqs[word]  # count(word, tag)
            total_w = sum(freqs.values())  # count(tag)

            # inverse log to respect the fact that weight is actually a score
            probab = -math.log(val / total_w)

            # write transition rule to file
            tagger_trans.write(f"0\t0\t{word}\t{tag}\t{probab}\n")

        # Now we handle the probabilities for an unknown word
        # <unk> can be tagged with any tag, with equal possibility
        unkprob = 1 / len(cfd.keys())

        for tag in cfd.keys():
            tagger_trans.write(f"0\t0\t<unk>\t{tag}\t{unkprob}\n")

        tagger_trans.write("0")  # funally write a 0 at the end of the file

    # finally we compile the file we just generated into a transducer
    call(
        f"fstcompile --isymbols=w/lex.syms --osymbols=w/lex.syms --keep_osymbols --keep_isymbols w/{name}.txt | " +
        f"fstarcsort > w/{name}.fsa")


def create_baseline_model():
    """
    Baseline model is just the word -> IOB tag transducer + unigram model with no smoothing
    """
    call("farcompilestrings --symbols=w/lex.syms --keep_symbols --unknown_symbol='<unk>' w/iob_ngram_file.txt | " + # use word -> IOB tag transducer
         "ngramcount --order=1 --require_symbols=false - | " + # unigram 
         "ngrammake --method=unsmoothed - | " + # with no smoothing
         
         # now we compose the tagger with the ngram model
         "fstcompose w/iob_tagger_trans.fsa - " +
         " > w/baseline_wfst_ngrm.fsa")


def create_iob_ngram_model(method="kneser_ney", order="3"):
    """
    Creates the transducer + ngram model. Only considering words and normal IOB tags
    """

    call(f"farcompilestrings --symbols=w/lex.syms --keep_symbols --unknown_symbol='<unk>' w/iob_ngram_file.txt | " +
         f"ngramcount --order={order} --require_symbols=false - | " +
         f"ngrammake --method={method} - | " +
         # now we compose the tagger with the ngram model
         f"fstcompose w/iob_tagger_trans.fsa - " +
         f" > w/iob_wfst_ngrm.fsa")


def create_iob_and_wrds_ngram_model(method="kneser_ney", order="4"):
    """
    Creates the transducer + ngram model.
    Consider IOB tags + words

    """

    call(f"farcompilestrings --symbols=w/lex.syms --keep_symbols --unknown_symbol='<unk>' w/iob_and_w_ngram_file.txt | " +
         f"ngramcount --order={order} --require_symbols=false - | " +
         f"ngrammake --method={method} - | " +
         # compose the tagger with the ngram model
         f"fstcompose w/iob_and_w_tagger_trans.fsa - " +
         f" > w/iob_and_w_wfst_ngrm.fsa")
