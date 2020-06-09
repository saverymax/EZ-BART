model_config=bart_config/with_question/bart-bin
model_path=checkpoints_bioasq_with_question/
QUESTION="What is question-driven summarization?"
prediction_path=predictions/sample_predictions.json
data_path=../data_processing/data/sample_data.json
echo Saving summaries to $prediction_path

python run_inference.py \
    --question="${QUESTION}" \
    --prediction_file=$prediction_path \
    --model_path=$model_path \
    --model_config=$model_config \
    --data=$data_path
