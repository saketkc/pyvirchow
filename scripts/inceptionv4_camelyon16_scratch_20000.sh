DATASET_DIR=/Z/personal-folders/interns/saket/tf-datasets/camelyon16-training-200000-images/
TRAIN_DIR=/Z/personal-folders/interns/saket/inceptionv4_train_logs_camelyon16_200000
CHECKPOINT_PATH=/Z/personal-folders/interns/saket/github/pywsi/checkpoint_models/inception_v4.ckpt
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_split_name=train \
    --model_name=inception_v4 \
    --batch_size=32 \
    --learning_rate=0.001 \
    --num_clones=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits \
    --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits
