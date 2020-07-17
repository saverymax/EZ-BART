"""
Module for classes to prepare validation dataset from MedInfo dataset.
Data format will be {key: {'question': question, 'summary':, summ, 'articles': articles} ...}

Additionally, format for question driven summarization. For example:
python prepare_validation_data.py -t --add-q
"""


import json
import re


class MedInfo():

    def _load_collection(self):
        """
        Load medinfo collection prepared in the process_medinfo.py script
        """
        with open("data/medinfo/medinfo_collection.json", "r", encoding="utf-8") as f:
            medinfo = json.load(f)

        return medinfo

    def save_section2answer_validation_data(self):
        """
        For questions that have a corresponding section-answer pair, save the
        validation data in following format 
        {'query': {'summary': summary, 'document': src text}}
        """
        dev_dict = {}
        medinfo = self._load_collection()
        data_pair = 0
        for i, question in enumerate(medinfo):
            try:
                # There may be multiple answers per question, but for the sake of the validation set,
                # just use the first answer
                if 'section_text' in medinfo[question][0]:
                    article = medinfo[question][0]['section_text']
                    summary = medinfo[question][0]['answer']
                    # Stripping of whitespace was done in processing script for section and full page
                    # but not for answer or question
                    summary = re.sub(r"\s+", " ", summary)
                    question = re.sub(r"\s+", " ", question)
                    assert len(summary) <= (len(article) + 10)
                    data_pair += 1
                    dev_dict[i] = {'query': question, 'summary': summary, 'document': article}
            except AssertionError:
                print("Answer longer than summary. Skipping element")

        print("Number of page-section pairs:", data_pair)

        with open("data/medinfo/medinfo_section2answer_validation_data.json", "w", encoding="utf-8") as f:
            json.dump(dev_dict, f, indent=4)


def process_data():
    """
    Main function for saving data
    """
    MedInfo().save_section2answer_validation_data()

if __name__ == "__main__":
    process_data()
