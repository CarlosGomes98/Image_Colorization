import datetime
import os, sys
import numpy as np
import tensorflow as tf
from skimage import io, color
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, Input, UpSampling2D, Conv2D, Conv1D, Dense, Dropout, BatchNormalization, Flatten, Conv2DTranspose, Reshape
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from DataGeneratorImages import DataGenerator
from tensorflow.python.client import device_lib
from model.utilities import parse_function

image_size = 128

if tf.test.gpu_device_name():
    print('Default GPU: {}'.format(tf.test.gpu_device_name()))
else:
    print('Failed to find default GPU.')
    sys.exit(1)

def parse_function(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, ratio=2, channels=3)
    image_lab = tf.py_func(rgb2lab_32, [image], tf.float32)
    image_L = tf.py_func(preprocess_and_return_X, [image_lab], tf.float32)
    image_L = tf.reshape(image_L, [128, 128, 1])
    return image_L

model = load_model(os.path.join("~/Image_Colorization/", sys.argv[1], sys.argv[2]))

model.compile(loss="categorical_crossentropy",
            optimizer="adam",
            metrics=['accuracy'])

os.mkdir(os.path.join("~/Image_Colorization/", sys.argv[1], "test"))

batch_size = 16

test = []
test_path = "~/Image_Colorization/data/Test"
for image in os.listdir(test_path)[:1000]
    test.append(os.path.join(test_path, image))

test_dataset = tf.data.Dataset.from_tensor_slices(partition["train"])
test_dataset = test_dataset.apply(tf.data.experimental.shuffle_and_repeat(len(partition["train"])))
test_dataset = test_dataset.map(parse_function, num_parallel_calls=8)
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(1)

validation_dataset = tf.data.Dataset.from_tensor_slices(partition["validation"])
validation_dataset = validation_dataset.shuffle(len(partition["validation"]))
validation_dataset = validation_dataset.map(parse_function, num_parallel_calls=4).repeat()
validation_dataset = validation_dataset.batch(batch_size)
validation_dataset = validation_dataset.prefetch(1)
buckets = np.load("model/pts_in_hull.npy")
rebalance = np.load("model/rebalance.npy")
# num_train_batches = 4039
num_train_batches = 258500//batch_size
num_validation_batches = 10000//batch_size
model.summary()

class WeightsSaver(Callback):
    def __init__(self, N, output_path):
        self.N = N
        self.output_path = output_path
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        if self.batch % self.N == 0 and self.N != 0:
            name = 'current_model.h5'
            self.model.save(name)
            # try:
            #     os.system('gsutil cp ' + name + ' ' + self.output_path)
            # except:
            #     print("Could not upload current model")
        self.batch += 1

checkpoint = ModelCheckpoint(os.path.join(self.output_path, "curr.hdf5"),
                            monitor="val_loss",
                            verbose=1,
                            save_weights_only = False,
                            save_best_only=False,
                            mode="auto",
                            period=1)

# every_epoch = WeightsSaver(num_train_batches, self.output_path)

# every_10 = ModelCheckpoint("latest.hdf5",
#                           monitor="accuracy",
#                           verbose=1,
#                           save_best_only=False,
#                           mode='auto',
#                           period=5)

tensorboard = TensorBoard(log_dir=self.output_path, histogram_freq=0, write_images=True)
callbacks = [tensorboard, checkpoint]

model.fit(train_dataset.make_one_shot_iterator(),
            validation_data = validation_dataset.make_one_shot_iterator(),
            callbacks=callbacks,
            steps_per_epoch=num_train_batches,
            validation_steps=num_validation_batches,
            epochs=2)

model.save(os.path.join(self.output_path, "model.h5"))

def set_up_and_train(self):
if self.model_path is None:
    model = self.set_up_model()
else :
    model = load_model(self.model_path)
self.train(model)
