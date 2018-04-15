import model


def preprocess_files():
    """
    Load the different dataset files and process them to the appropiate format: [ ... (word, tag) ... ]
    """

    #? process files and create lexicon
    train_file = open("../P1_data/data/NLSPARQL.train.data",
                      "r", encoding="utf8").read()

    test_file = open("../P1_data/data/NLSPARQL.test.data",
                     "r", encoding="utf8").read()

    train_set = [[pair.split("\t")
                  for pair in sentence.split("\n")]
                 for sentence in train_file.split("\n\n")]

    def imp_extract_w_and_tag(pair): 
        """
        For improved version, the O tags are replaced with
        O__word
        """
        word = pair[0]
        tag = pair[1]

        # word is appended to the tag
        if tag == "O":
            tag = tag + "__" + word

        return word, tag

    # We want improved train set to be of the form [word, iob_tag__word if tag is O else iob_tag]
    train_wrd_tag = [[imp_extract_w_and_tag(pair)
                      for pair in sentence]
                     for sentence in train_set]

    ##############
    # Create lexicon
    # Lex will enclude all words in training + IOB tags + IOB tags with words (+ eps and unk tokens)
    words = [item for sentence in train_set
             for pair in sentence
             for item in pair] \
        + \
            [item for sentence in train_wrd_tag
                for pair in sentence
                for item in pair]

    lex = sorted(list(set(words)))
    lex = ["<epsilon>"] + lex + ["<unk>"]

    f_lex = open("w/lex.syms", "w")
    for i, w in enumerate(lex):
        f_lex.write(f"{w}\t{i}\n")

    f_lex.close()
    ################

    test_set = [[pair.split("\t")
                 for pair in sentence.split("\n")]
                for sentence in test_file.split("\n\n")]

    test_set = [[[word if word in lex else "<unk>", tag] # replace any unknown words in the test set with the <unk> keyword
                 for word, tag in sentence]
                for sentence in test_set]

    ################
    # Create n-gram files
    #
    # here we write a file that for each line contains the tags in a sentence of the training set, separated by a space
    # we will later process these files with OpenGrm to create the ngram models
    with open("w/iob_ngram_file.txt", "w") as ngram_file:
        content = [" ".join(tag for _, tag in sentence)
                   for sentence in train_set]
        content = "\n".join(content)
        ngram_file.write(content)

    # create the same ngram file but with the improved training set
    with open("w/iob_and_w_ngram_file.txt", "w") as ngram_file:
        content = [" ".join(tag for _, tag in sentence)
                   for sentence in train_wrd_tag]
        content = "\n".join(content)
        ngram_file.write(content)

    # finally return the processed train sets (basic and improved), test set and lexicon file
    return train_set, train_wrd_tag, test_set, lex 
