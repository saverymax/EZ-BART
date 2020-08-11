import json

with open("section2answer_single_abstractive_summ.json", "r", encoding="utf-8") as f:
    mediqa_abs = json.load(f)

with open("section2answer_single_extractive_summ.json", "r", encoding="utf-8") as f:
    mediqa_ext = json.load(f)

ez_mediqa_abs = {}
ez_mediqa_ext = {}
for answer in mediqa_abs:
    ez_mediqa_abs[answer] = {}
    ez_mediqa_abs[answer]['query'] = mediqa_abs[answer]['question']
    ez_mediqa_abs[answer]['document'] = mediqa_abs[answer]['articles']
    ez_mediqa_abs[answer]['summary'] = mediqa_abs[answer]['summary']

for answer in mediqa_ext:
    ez_mediqa_ext[answer] = {}
    ez_mediqa_ext[answer]['query'] = mediqa_ext[answer]['question']
    ez_mediqa_ext[answer]['document'] = mediqa_ext[answer]['articles']
    
with open("mediqa_abs_for_ezbart.json", "w", encoding="utf-8") as f:
    json.dump(ez_mediqa_abs, f, indent=4)

with open("mediqa_ext_for_ezbart.json", "w", encoding="utf-8") as f:
    json.dump(ez_mediqa_ext, f, indent=4)
    
    

