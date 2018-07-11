DATASET_DIR=/Z/personal-folders/interns/saket/tf-datasets/camelyon16-training-200000-images/
TRAIN_DIR=/Z/personal-folders/interns/saket/inceptionv4_train_logs_camelyon16_200000
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=12 \
    --num_clones=1 \
    --model_name=inception_v4
