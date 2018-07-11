DATASET_DIR=/Z/personal-folders/interns/saket/tf-datasets/camelyon16-testing/
CHECKPOINT_FILE=/Z/personal-folders/interns/saket/inceptionv4_train_logs_camelyon16
python eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_FILE} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_split_name=validation \
    --model_name=inception_v4

