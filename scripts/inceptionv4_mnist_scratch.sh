DATASET_DIR=/Z/personal-folders/interns/saket/tf-datasets/mnist/
TRAIN_DIR=/Z/personal-folders/interns/saket/inceptionv4_train_logs_mnist
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=mnist \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v4
