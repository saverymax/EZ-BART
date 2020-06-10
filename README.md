# EZ BART Summarization Tool
This is the repository for the BART summarization tool discussed in the paper *Question-Driven Summarization of Answers to Consumer Health Questions*. 

## Installation
This installation assumes Python >= 3.6 and anaconda or miniconda is installed.   
Note that this install works *only* for running inference. For training, different installation instructions will be provided.
```
git clone https://github.com/saverymax/EZ-BART.git
cd EZ-BART
conda create --name ez_bart python=3.7 pytorch torchvision cpuonly -c pytorch
conda activate ez_bart
pip install -r requirements.txt
cd bart
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

## Inference
If you are using bash, a sample script for running inference is provided in the bart directory.
```
bash run_inference.sh
```
Or if you'd rather run the python command directly:
```
python run_inference.py --question="Do I have COVID-19?" --prediction_file=predictions/bart_summs.json --model_path=checkpoints_bioasq_with_question --model_config=bart_config/with_question/bart-bin --data=../data_processing/data/sample_data.json
```
The script assumes the input file is in the following json format:
```
{
"<UNIQUE ARTICLE ID 1>": "This is the text of the article to be summarized",
...
"<UNIQUE ARTICLE ID n>": "This is the text of another article to be summarized",
}
```
See data_processing/data/sample_data.json for a test case.

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

