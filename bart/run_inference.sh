model_config=bart_config/with_question/bart-bin
model_path=checkpoints_bioasq_with_question/
QUESTION="Do I have COVID-19?"
joined_question=${QUESTION// /-}
joined_question=${joined_question::-1}
prediction_path=predictions/bart_summs_${joined_question}.json
data_path=../data_processing/data/sample_data.json
echo $prediction_path

python run_inference.py \
    --question="${QUESTION}" \
    --prediction_file=$prediction_path \
    --model_path=$model_path \
    --model_config=$model_config \
    --data=$data_path
