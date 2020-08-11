import json
import re

with open("cochrane_clinical_answer_train_collection.json", "r", encoding="utf-8") as f:
    train = json.load(f)

with open("cochrane_clinical_answer_val_collection.json", "r", encoding="utf-8") as f:
    val = json.load(f)

ez_train = {}
ez_val = {}
t_cnt = 0
v_cnt = 0
for answer in train:
    t_cnt += 1
    ez_train[answer] = {}
    question = train[answer]['question'].strip()
    question = re.sub(r"\s+", " ", question)
    ez_train[answer]['query'] = question
    document = train[answer]['article']['background'].strip()
    document = re.sub(r"\s+", " ", document)
    ez_train[answer]['document'] = document
    summary = train[answer]['answer'].strip()
    summary = re.sub(r"\s+", " ", summary)
    ez_train[answer]['summary'] = summary

for answer in val:
    v_cnt += 1
    ez_val[answer] = {}
    question = val[answer]['question'].strip()
    question = re.sub(r"\s+", " ", question)
    ez_val[answer]['query'] = question
    document = val[answer]['article']['background'].strip()
    document = re.sub(r"\s+", " ", document)
    ez_val[answer]['document'] = document
    summary = val[answer]['answer'].strip()
    summary = re.sub(r"\s+", " ", summary)
    ez_val[answer]['summary'] = summary

with open("clinical_answer_train_for_ezbart.json", "w", encoding="utf-8") as f:
    json.dump(ez_train, f, indent=4)

with open("clinical_answer_val_for_ezbart.json", "w", encoding="utf-8") as f:
    json.dump(ez_val, f, indent=4)
print("Train size:", t_cnt)
print("Val size:", v_cnt)
