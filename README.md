# EZ BART Summarization Tool

# Installation
```
conda create --name ez_bart pytorch torchvision -c pytorch
pip install -r requirements.txt
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

# Inference
```
bash run_inference.sh
```
An example of running inference is provided in the run_inference.sh script.

The script assumes the input file is in the following json format:
```
{
"<UNIQUE ARTICLE ID 1>": "This is the text of the article to be summarized",
...
"<UNIQUE ARTICLE ID n>": "This is the text of another article to be summarized",
}
```

# Training ***Implementation in progress***
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

