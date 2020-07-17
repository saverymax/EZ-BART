# EZ BART Summarization Tool
This is the repository for the BART summarization tool discussed in the paper *Question-Driven Summarization of Answers to Consumer Health Questions*.

## Installation
This installation assumes Python >= 3.6 and anaconda or miniconda is installed.   
To install anaconda, see instructions here https://docs.anaconda.com/anaconda/install/   
EZ-BART will run on either Windows or Linux, but the installation for anaconda varies per platform.
For windows, you can download the installer [here](https://docs.anaconda.com/anaconda/install/windows/) and follow the prompts.

Note that this install works *only* for running inference. For training, see below.
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
**--model_config** The path to the configuration for the model. If using the fine-tuned weights, this is optional to include since the default specifies the correct path. The path will need to be adjusted if you are fine-tuning the weights yourself.   


## Training
As described above, the fine-tuned BART weights can be used as-is. However, should you want to use weights fine-tuned on data other than BioASQ, the instructions here demonstrate how to retrain the BART model with your own data, or on a selection of biomedical datasets included in this repository.

### Environment
A GPU with 32gb of VRAM is required to train BART. This memory requirement assumes a maximum source document sequence length of 1024 subword tokens, and a batch size of 8. You can feasibly use a 16gb GPU with a smaller sequence length instead, and will not suffer performance degredation to a large degree with a maximum sequence length of 512.

Let's create a new environment for training:
```
conda create --name ez_bart_train python=3.7 pytorch torchvision cudatoolkit -c pytorch
conda activate ez_bart_train
pip install -r requirements.txt
cd bart
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```
Note that the GPU version of pytorch will be installed, with the CUDA tool kit. It is also possible to install Nvidia's Apex library for mixed precision training, but as this install depends heavily on your individual system, we do not include it in the instructions here. See https://github.com/saverymax/qdriven-chiqa-summarization or https://github.com/pytorch/fairseq for instructions on an install that *might* work for you.


### Data

Using your own training data is encouraged, but because of the unique content and structure of each dataset, it may require custom processing on your end. Once your dataset has been formatted in the following structure, you can train with it after one more fairseq-specific processing step.  

First make sure your data is in the correct json format, with 'query', 'source_document', and 'summary' keys.
```
{
'example_id_1': {'query': "query text", 'source_document': "document to be summarized", 'summary': "summary of the document},
...
'example_id_n': {'query': "query text", 'source_document': "document to be summarized", 'summary': "summary of the document},
}
```
It is not necessary that the value of the query key be a query per se; you may use a topic or keyword to focus the content of the summary. It is also possible to not include query key in the data, but make sure to leave out the --add-q option in the processing command.

For futher information regarding processing your dataset, a working example of processing the BioASQ and Medinfo data is included in the data_processing directory, in process_bioasq.py and process_medinfo.py

Then, to prepare data files for BART, run the following. This formats the data to so that fairseq can efficiently process it during training.
```
your data stuff
```

For example
```
python -m data_processing.prepare_training_data --train_path=data_processing/data/bioasq/bioasq_collection.json --val_path=data_processing/data/medinfo/medinfo_section2answer_validation_data.json --config_path=/data/saveryme/ez_bart_config --add_q

bash data_processing/make_bart_data.sh
```
This will tokenize and prepare the training and validation data for the model to efficiently process. The script will download encoder.json, vocab.bpe, and dict.txt from a Facebook cloud service.

Now you are ready to train!
```
BART_CONFIG=/path/to/your/ez_bart_config
CHECKPOINT_DIR=/path/to/save/bart_checkpoints

CUDA_VISIBLE_DEVICES=0 python -m bart.train $BART_CONFIG/bart-bin \
    --save-dir=$CHECKPOINT_DIR \
    --restore-file bart/bart.large/model.pt \
    --max-tokens 1024 \
    --truncate-source \
    --task translation \
    --source-lang source --target-lang target \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr 3e-05 --total-num-update 20000 --warmup-updates 500 \
    --update-freq 16 \
    --skip-invalid-size-inputs-valid-test \
    --ddp-backend=no_c10d \
    --keep-last-epochs=2 \
    --find-unused-parameters;
```
If you have a environment suitable for mixed precision training, include the ```--fp16``` option as well.

Once your model is trained, you can use it for inference as described in the first section. Just make sure to specify the path to your new weights and config.
For example
```
python -m bart.run_inference --question="Is my BART model trained properly?" --prediction_file=bart/predictions/bart_summs.json --model_path=/path/to/your/new_checkpoint --data=data_processing/data/sample_data.json --model_config=/your/new/ez_bart_config/bart-bin
```

## FAQ
1. When is my model done training?
  This depends on the size of your dataset and difficulty of the task. We observed good results after only a few epochs. It may serve you best to run training for different training times and compare the results.
2. Does the fairseq processing support other tokenizers and data processors?
  Yes! However, you will have to look further into the implementation details described within fairseq itself.
3. Can I use Huggingface's version of BART instead?
  Certainly, however, we have not thoroughly tested the reproducibility of the Huggingface training implementation compared to the original BART code. That is why the fairseq library is used here, instead of the easier to manage Huggingface. Additionally, fairseq handles batching, sequence lengths, and multi-gpu training more efficiently.
4. I am having trouble getting my data into the correct format. I'm getting some weird errors...
  Data processing is often the most difficult part of the deep learning pipeline. The processing script provided here only takes into account a limited number of use-cases. If you are experiencing errors, there is a good chance your data breaks something in this pipeline. If this is the case, please create an issue and we'll see what we can do.
5. I am running out of memory.
  Simplest option: Decrease the max-tokens argument, which specifies the maximum number of tokens in the batch. You may also may decrease the length of --max-source-positions (default 1024), which will decrease the maximum length of a single example. Finally, you may consider using mixed precision training. See issues at fairseq https://github.com/pytorch/fairseq/issues/1413 and https://github.com/pytorch/fairseq/issues/1818 for more context on the way fairseq handles batching and multi-gpu training.

Happy summarization!
