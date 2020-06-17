# EZ BART Summarization Tool
This is the repository for the BART summarization tool discussed in the paper *Question-Driven Summarization of Answers to Consumer Health Questions*. 

## Installation
This installation assumes Python >= 3.6 and anaconda or miniconda is installed.   
To install anaconda, see instructions here https://docs.anaconda.com/anaconda/install/   
EZ-BART will run on either Windows or Linux, but the installation for anaconda varies per platform.
For windows, you can download the installer [here](https://www.anaconda.com/download/#windows) and follow the prompts.

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
FOr inference, you will need to download the fine-tuned BART weights, and you will need data formatted properly.  
The weights can be downloaded at https://bionlp.nlm.nih.gov/bart_finetuned_checkpoint.zip. Once they are unzipped, you can specify the path to them as shown below.   
The data format expected by the inference script is shown here:
```
{
"<UNIQUE ARTICLE ID 1>": "This is the text of the article to be summarized",
...
"<UNIQUE ARTICLE ID n>": "This is the text of another article to be summarized",
}
```
See data_processing/data/sample_data.json for an example.

Once your data is in the correct format, you are ready to summarize!
For example, from the base directory of this repository (EZ-BART), using the provided sample data you can try running
```
python -m bart.run_inference --question="Do I have COVID-19?" --prediction_file=bart/predictions/bart_summs.json --model_path=bart/checkpoints_bioasq_with_question --data=data_processing/data/sample_data.json
```
### FLAGS
**--question** The question you would like to drive the content of the summary.   
**--prediction_file** The path of the file you would like the summaries saved to.   
**--model_path** The path to the downloaded BART weights.   
**--data** The path to the json with the data you would like to be summarized.   


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

