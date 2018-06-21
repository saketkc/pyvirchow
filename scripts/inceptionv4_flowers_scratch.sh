DATASET_DIR=/Z/personal-folders/interns/saket/tf-datasets/flowers/
TRAIN_DIR=/Z/personal-folders/interns/saket/inceptionv4_train_logs_flowers
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=12 \
    --num_clones=2 \
    --model_name=inception_v4
