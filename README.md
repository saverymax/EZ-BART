# EZ BART Summarization Tool
This is the repository for the BART summarization tool discussed in the paper *Question-Driven Summarization of Answers to Consumer Health Questions*

## Installation
```
conda create --name ez_bart pytorch torchvision -c pytorch
pip install -r requirements.txt
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

## Inference
If you are using bash, a sample script for running inference is provided.
```
bash run_inference.sh
```
Or if you'd rather just run the python command directly:
```
python run_inference.py \
    --question="Do I have COVID-19?" \
    --prediction_file=predictions/bart_summs.json \
    --model_path=checkpoints_bioasq_with_question \
    --model_config=bart_config/with_question/bart-bin \
    --data=../data_processing/data/sample_data.json
```
The script assumes the input file is in the following json format:
```
{
"<UNIQUE ARTICLE ID 1>": "This is the text of the article to be summarized",
...
"<UNIQUE ARTICLE ID n>": "This is the text of another article to be summarized",
}
```
See the ../data_processing/data/sample_data.json file for a test case.

## Training *Implementation in progress*
The fine-tuned BART weights are provided with the release of this code. However, if you are interested in retraining the model with fairseq, instructions are provided here.
```
wget https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz
tar -xzf bart.large.tar.gz
bash prepare_training_data.sh
```
This will process the bioasq and medinfo data for training and validation, respectively.   
Then, to prepare data files for BART:
```
bash make_bart_data.sh
```

