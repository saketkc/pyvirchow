# Check point models
wget -c http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
wget -c http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz

# Retraining InceptionV4 on Flowers

```bash
DATASET_DIR=/Z/personal-folders/interns/saket/tf-datasets/mnist/
TRAIN_DIR=/Z/personal-folders/interns/saket/inceptionv4_train_logs_mnist
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=mnist \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=inception_v4
```


# Misc Notes about optimizing run time

1. We have two gpus, so make sure both are being used.

1A. watch -n 3 nvidia-smi
	This step should show close to 100% GPU utilization.

1B. Training should ideally use number of clones = number of GPUs.
	
1C. I got a gailure trying to run with number of batches=32 originally. 12 worked. 
A higher batch size is generally better, but is also more GPU heavy

Here's a copy pasted version from Google's [instructions](https://github.com/tensorflow/models/tree/master/research/inception#adjusting-memory-demands)
# Adjusting Memory Demands

Training this model has large memory demands in terms of the CPU and GPU. Let's discuss each item in turn.

GPU memory is relatively small compared to CPU memory. 
Two items dictate the amount of GPU memory employed -- model architecture and batch size. Assuming that you keep the model architecture fixed, the sole parameter governing the GPU demand is the batch size. A good rule of thumb is to try employ as large of batch size as will fit on the GPU.

If you run out of GPU memory, either lower the --batch_size or employ more GPUs on your desktop. The model performs batch-splitting across GPUs, thus N GPUs can handle N times the batch size of 1 GPU.

The model requires a large amount of CPU memory as well. We have tuned the model to employ about ~20GB of CPU memory. Thus, having access to about 40 GB of CPU memory would be ideal.

If that is not possible, you can tune down the memory demands of the model via lowering --input_queue_memory_factor. Images are preprocessed asynchronously with respect to the main training across --num_preprocess_threads threads. The preprocessed images are stored in shuffling queue in which each GPU performs a dequeue operation in order to receive a batch_size worth of images.

In order to guarantee good shuffling across the data, we maintain a large shuffling queue of 1024 x input_queue_memory_factor images. For the current model architecture, this corresponds to about 4GB of CPU memory. You may lower input_queue_memory_factor in order to decrease the memory footprint. Keep in mind though that lowering this value drastically may result in a model with slightly lower predictive accuracy when training from scratch. Please see comments in image_processing.py for more details.
	
