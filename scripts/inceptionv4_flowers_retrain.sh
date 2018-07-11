DATASET_DIR=/Z/personal-folders/interns/saket/tf-datasets/flowers/
TRAIN_DIR=/Z/personal-folders/interns/saket/inceptionv4_retrain_logs_flowers
CHECKPOINT_PATH=/Z/personal-folders/interns/saket/github/pywsi/checkpoint_models/inception_v4.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --batch_size=32 \
    --num_clones=2 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits
