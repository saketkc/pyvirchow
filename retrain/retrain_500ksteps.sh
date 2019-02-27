#!/bin/bash
python retrain.py --image_dir /mnt/disks/data/images-for-retrain/training/ \
    --train_batch_size  32\
    --validation_percentage 10\
    --validation_batch_size 5000\
    --how_many_training_steps 500000\
    --learning_rate 1e-6\
    --tfhub_module https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1\
    --summaries_dir /mnt/disks/data/retrain-logs-all-train_inceptionv3_500k \
    --output_labels /mnt/disks/data/retrain-logs-all-train_inceptionv3_500k/output_label.txt \
    --output_graph /mnt/disks/data/retrain-logs-all-train_inceptionv3_500k/output_graph.pb \
    --bottleneck_dir /mnt/disks/data/retrain-logs-all-train_inceptionv3-bottleneck \
    --intermediate_output_graphs_dir /mnt/disks/data/retrain-logs-all-train_inceptionv3-intermediate_500k \
    --print_misclassified_test_images
#--validation_batch_size=-1
