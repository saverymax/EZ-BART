"""
Module for classes to prepare training and validation datasets
"""

import json
import argparse
import re
import os

from sklearn.utils import shuffle as sk_shuffle


def get_args():
    """
    Argument defnitions
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--train_path",
                        dest="train_path",
                        default="data/bioasq/bioasq_collection.json",
                        help="Path to the training data collection")
    parser.add_argument("--val_path",
                        dest="val_path",
                        default="data/medinfo/medinfo_section2answer_validation_data.json",
                        help="Path to the validation data collection")
    parser.add_argument("--add_q",
                        dest="add_q",
                        action="store_true",
                        help="Concatenate the question to the beginning of the text.")
    parser.add_argument("--config_path",
                        dest="config_path",
                        default="",
                        help="Path for the config created by this script")
    return parser


class ProcessTrain():
    """
    Class to create training and validation datasets for BART
    Created using reference to fairseq processing  code 
    www.stuff.com
    """

    def create_data_for_bart(self):
        """
        Write the train and val data to file so that the processor and tokenizer for bart will read it, as per fairseqs design
        """
        Q_END = " [QUESTION?] "
        
        if not os.path.exists(args.train_path) or not os.path.exists(args.val_path):
            raise IOError("Please provide a valid path to train and validation collections")

        with open(args.train_path, "r", encoding="utf8") as f:
            train_collection = json.load(f)

        with open(args.val_path, "r", encoding="utf-8") as f:
            val_collection = json.load(f)

        try:
            print("Creating config directory:", args.config_path)
            os.makedirs(args.config_path) 
        except FileExistsError:
            print("Directory ", args.config_path , " already exists. Make sure you are not writing over existing config data")
            raise 

        train_src = open("{c}/bart.train.source".format(c=args.config_path), "w", encoding="utf8")
        train_tgt = open("{c}/bart.train.target".format(c=args.config_path), "w", encoding="utf8")
        val_src = open("{c}/bart.val.source".format(c=args.config_path), "w", encoding="utf8")
        val_tgt = open("{c}/bart.val.target".format(c=args.config_path), "w", encoding="utf8")

        train_summaries = []
        train_docs = []
        for ex in train_collection:
            summ = train_collection[ex]['summary'].strip()
            doc = train_collection[ex]['document'].strip()
            if 'query' in train_collection[ex]:
                query = train_collection[ex]['query'].strip()
                doc = query + Q_END + doc
            train_docs.append(doc)
            train_summaries.append(summ)

        val_summaries = []
        val_docs = []
        for ex in val_collection:
            summ = val_collection[ex]['summary'].strip()
            doc = val_collection[ex]['document'].strip()
            if 'query' in val_collection[ex]:
                query = val_collection[ex]['query'].strip()
                doc = query + Q_END + doc
            val_docs.append(doc)
            val_summaries.append(summ)

        snp_cnt = 0
        print("Shuffling data")
        # Shuffle summary doc pairs
        train_summaries, train_docs = sk_shuffle(train_summaries, train_docs, random_state=13)
        for summ, doc in zip(train_summaries, train_docs):
            snp_cnt += 1
            train_src.write("{}\n".format(doc))
            train_tgt.write("{}\n".format(summ))


        val_summaries, val_docs = sk_shuffle(val_summaries, val_docs, random_state=13)
        for summ, doc in zip(val_summaries, val_docs):
            val_src.write("{}\n".format(doc))
            val_tgt.write("{}\n".format(summ))

        train_src.close()
        train_tgt.close()
        val_src.close()
        val_tgt.close()

        # Original config is stored in ../bart/bart_config/. If overwritten can be downloaded from the github for this repo
        # Make sure there were no funny newlines added
        train_src = open("{c}/bart.train.source".format(c=args.config_path), "r", encoding="utf8").readlines()
        train_tgt = open("{c}/bart.train.target".format(c=args.config_path), "r", encoding="utf8").readlines()
        val_src = open("{c}/bart.val.source".format(c=args.config_path), "r", encoding="utf8").readlines()
        val_tgt = open("{c}/bart.val.target".format(c=args.config_path), "r", encoding="utf8").readlines()
        assert len(train_src) == snp_cnt, len(train_src)
        assert len(train_tgt) == snp_cnt
        assert len(val_src) == len(val_docs)
        assert len(val_tgt) == len(val_summaries)
        print("Number of train examples: ", len(train_src))
        print("Number of val examples: ", len(val_src))
        print("Assertions passed")


if __name__ == "__main__":
    global args
    args = get_args().parse_args()
    ProcessTrain().create_data_for_bart()
