DATASET_DIR=/Z/personal-folders/interns/saket/tf-datasets/flowers/
CHECKPOINT_FILE=/Z/personal-folders/interns/saket/inceptionv4_retrain_logs_flowers
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=validation \
    --model_name=inception_v4

