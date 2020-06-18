# EZ BART Summarization Tool
This is the repository for the BART summarization tool discussed in the paper *Question-Driven Summarization of Answers to Consumer Health Questions*. 

## Installation
This installation assumes Python >= 3.6 and anaconda or miniconda is installed.   
To install anaconda, see instructions here https://docs.anaconda.com/anaconda/install/   
EZ-BART will run on either Windows or Linux, but the installation for anaconda varies per platform.
For windows, you can download the installer [here](https://docs.anaconda.com/anaconda/install/windows/) and follow the prompts.

Note that this install works *only* for running inference. For training, installation instructions will be provided soon.
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
The data format expected by the inference script is shown here:
```
{
"<UNIQUE ARTICLE ID 1>": "This is the text of the article to be summarized",
...
"<UNIQUE ARTICLE ID n>": "This is the text of another article to be summarized",
}
```
See data_processing/data/sample_data.json for an example.

To run the model, first download the fine-tuned BART weights from https://bionlp.nlm.nih.gov/bart_finetuned_checkpoint.zip. Once they are unzipped, you can specify the path to them as shown in the example below.   

Once your data is in the correct format and you have downloaded the model, you are ready to summarize!
Activate your environment if it is not already activated
```
conda activate ez_bart
```
And then from the base directory of this repository (EZ-BART), you can try running
```
python -m bart.run_inference --question="Do I have COVID-19?" --prediction_file=bart/predictions/bart_summs.json --model_path=bart/bart_finetuned_checkpoint/checkpoints_bioasq_with_question --data=data_processing/data/sample_data.json --model_config=bart/bart_config/with_question/bart-bin
```
This example assumes you have stored the downloaded weights (bart_finedted_checkpoint/checkpoints_bioasq_with_question/checkpoint_best.pt) in the EZ-BART/bart directory, but you can store the weights wherever is convenient for your system.

### FLAGS
**--question** The question you would like to drive the content of the summary.   
**--prediction_file** The path of the file you would like the summaries saved to.   
**--model_path** The path to the downloaded BART weights. Include only the path to the directory the model is stored in, not the .pt file itself.   
**--data** The path to the json with the data you would like to be summarized.   
**--model_config** The path to the configuration for the model. If using the pretrained weights, this is optional to include since the default specifies the correct path. The path will need to be adjusted if you are fine-tuning the weights yourself.   


## Training 
*Implementation in progress*
The fine-tuned BART weights are provided with the release of this code. A HuggingFace (https://huggingface.co/) training protocol will be provided in the future.
```
bash prepare_training_data.sh
```
This will process the bioasq and medinfo data for training and validation, respectively.   
Then, to prepare data files for BART:
```
bash make_bart_data.sh
```
Then
```
python -m EZ-BART.run_hf_train --model_config=path/to/config --model_checkpoint=path_to_checkpoint --training_data=path/to/data --hyperparams=some_hyperparams
tar -xzf bart.large.tar.gz
```

