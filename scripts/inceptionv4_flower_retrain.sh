DATASET_DIR=/Z/personal-folders/interns/saket/tf-datasets/flower
TRAIN_DIR=/tmp/flowers-models/inception_v4
CHECKPOINT_PATH=/Z/personal-folder/interns/saket/github/pywsi/checkpoints/inception_v4.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits
