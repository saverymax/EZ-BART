"""
Script for BART inference.
Modified from fairseq example CNN-dm script
"""

import argparse
import json
import glob
import time
import os

from tqdm import tqdm
import torch
from fairseq.models.bart import BARTModel


def get_args():
    """
    Argument defnitions
    """
    parser = argparse.ArgumentParser(description="Arguments for data exploration")
    parser.add_argument("--prediction_file",
                        dest="prediction_file",
                        help="File to save predictions")
    parser.add_argument("--question",
                        dest="question_text",
                        help="The text of the question")
    parser.add_argument("--model_path",
                        dest="model_path",
                        help="Path to model checkpoints")
    parser.add_argument("--batch_size",
                        dest="batch_size",
                        default=32,
                        help="Batch size for inference")
    parser.add_argument("--model_config",
                        dest="model_config",
                        default="bart/bart_config/with_question/bart-bin",
                        help="Path to model config for BART.")
    parser.add_argument("--data",
                        dest="data_path",
                        default="./",
                        help="Path to input data for summarization")
    return parser


def main():
    """
    Main function for running inference on given input text
    """
    args = get_args().parse_args()
    start_time = time.time()

    if not os.path.exists(args.model_path):
        raise IOError("Path to model does not exist. Please specify the path to the model without including the .pt file itself.")

    if not os.path.exists(args.model_config):
        raise IOError("Path to model config does not exist. Please specify the path to the model configuration. By default, this is bart/bart_config/with_question/bart-bin")

    bart = BARTModel.from_pretrained(
        args.model_path,
        checkpoint_file='checkpoint_best.pt',
        data_name_or_path=args.model_config
    )

    bart.eval()
    gen_summaries = []
    articles = []
    ids = []
    QUESTION_END = " [QUESTION?] "
    batch_cnt = 0

    try:
        with open(args.data_path, 'r', encoding="utf-8") as f:
            src_text = json.load(f)
    except Exception as e:
        raise
        
    # Question driven
    for article_id in tqdm(src_text):
        ids.append(article_id)
        article = src_text[article_id]
        article = args.question_text + QUESTION_END + article
        articles.append(article)
        # Once the article list fills up, run a batch
        if len(articles) == args.batch_size:
            batch_cnt += 1
            print("Running batch {}".format(batch_cnt))
            # Hyperparameters as recommended here: https://github.com/pytorch/fairseq/issues/1364
            with torch.no_grad():
                predictions = bart.sample(articles, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
            for pred in predictions:
                #print(pred)
                gen_summaries.append(pred)
            articles = []

    if len(articles) != 0: 
        print("Predicting final batch...")
        predictions = bart.sample(articles, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        for pred in predictions:
            gen_summaries.append(pred)

    assert len(ids) == len(gen_summaries)
    prediction_dict = {}
    for i, s in zip(ids, gen_summaries):
        prediction_dict[i] = s

    with open(args.prediction_file, "w", encoding="utf-8") as f:
        json.dump(prediction_dict, f, indent=4)

    print("Done generating summaries. Run time:", time.time() - start_time)


if __name__ == "__main__":
    main()
