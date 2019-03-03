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
from utilities import parse_image, bucketize_image, augment_image
print(device_lib.list_local_devices())

image_size = 128

if tf.test.gpu_device_name():
    print('Default GPU: {}'.format(tf.test.gpu_device_name()))
else:
    print('Failed to find default GPU.')
    sys.exit(1)


def convLayer(input, filters, kernel_size, dilation=1, stride=1):
        return Conv2D(filters, kernel_size, padding="same", activation="relu", dilation_rate=dilation, strides=stride)(input)

class model:
    def __init__(self, image_path, output_path, model_path):
        self.image_path = image_path
        self.output_path = output_path
        self.model_path = model_path
        self.current_epoch = 1
        print("Image path: " + image_path)
        print("Output path: " + output_path)

    def get_epoch(self):
        self.current_epoch = self.current_epoch + 1
        return self.current_epoch

    def set_up_model(self):
        input_shape = (image_size, image_size, 1)

        model_input = Input(shape = input_shape)

        # conv1
        model_output = convLayer(model_input, 64, (3, 3))
        model_output = convLayer(model_output, 64, (3, 3), stride=2)
        model_output = BatchNormalization()(model_output)
        # conv2
        model_output = convLayer(model_output, 128, (3, 3))
        model_output = convLayer(model_output, 128, (3, 3), stride=2)
        model_output = BatchNormalization()(model_output)
        # conv3
        # model_output = convLayer(model_output, 256, (3, 3))
        model_output = convLayer(model_output, 256, (3, 3))
        model_output = convLayer(model_output, 256, (3, 3), stride=2)
        model_output = BatchNormalization()(model_output)
        # conv4
        # model_output = convLayer(model_output, 512, (3, 3))
        model_output = convLayer(model_output, 512, (3, 3))
        model_output = convLayer(model_output, 512, (3, 3))
        model_output = BatchNormalization()(model_output)
        # conv5
        # model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        model_output = BatchNormalization()(model_output)
        # conv6
        # model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        model_output = BatchNormalization()(model_output)
        # conv7
        # model_output = convLayer(model_output, 256, (3, 3))
        model_output = convLayer(model_output, 256, (3, 3))
        model_output = convLayer(model_output, 256, (3, 3))
        model_output = BatchNormalization()(model_output)
        # conv8
        model_output = UpSampling2D((2, 2))(model_output)
        model_output = convLayer(model_output, 256, (3, 3))
        model_output = UpSampling2D((2, 2))(model_output)
        model_output = convLayer(model_output, 256, (3, 3))
        model_output = UpSampling2D((2, 2))(model_output)
        model_output = convLayer(model_output, 256, (3, 3))

        # unary prediction
        model_output = Conv2D(313, (1, 1), activation="relu", padding="same")(model_output)
        model_output = Reshape((image_size*image_size, 313))(model_output)
        model_output = Activation("softmax")(model_output)
        model = Model(inputs=model_input, outputs=model_output)

        model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=['accuracy'])

        return model

    def train(self, model):
        self.output_path = self.output_path+"/"+datetime.datetime.now().strftime("%Y-%m-%d--%Hh%Mm")
        os.mkdir(self.output_path)
        train_data_path = os.path.join(self.image_path, "flowers", "flowers")
        validation_data_path = os.path.join(self.image_path, "flowers_val", "flowers_val")

        batch_size = 16

        partition = {"train": [], "validation": []}
        for image in os.listdir(train_data_path):
            partition["train"].append(os.path.join(train_data_path, image))

        for image in os.listdir(validation_data_path):
            partition["validation"].append(os.path.join(validation_data_path, image))

        train_dataset = tf.data.Dataset.from_tensor_slices(partition["train"])
        train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(len(partition["train"])))
        train_dataset = train_dataset.map(parse_image, num_parallel_calls=8)
        train_dataset = train_dataset.map(augment_image, num_parallel_calls=8)
        train_dataset = train_dataset.map(bucketize_image, num_parallel_calls=8)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(1)

        validation_dataset = tf.data.Dataset.from_tensor_slices(partition["validation"])
        validation_dataset = validation_dataset.apply(tf.data.experimental.shuffle_and_repeat(len(partition["validation"])))
        validation_dataset = validation_dataset.map(parse_image, num_parallel_calls=8)
        validation_dataset = validation_dataset.map(bucketize_image, num_parallel_calls=8)
        validation_dataset = validation_dataset.batch(batch_size)
        validation_dataset = validation_dataset.prefetch(1)

        buckets = np.load("model/pts_in_hull.npy")
        # rebalance = np.load("model/rebalance.npy")
        # num_train_batches = 4039
        # num_train_batches = 258500//batch_size
        # num_validation_batches = 10000//batch_size
        num_train_batches = 7189//batch_size
        num_validation_batches = 1000//batch_size
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
        
        checkpoint_best = ModelCheckpoint(os.path.join(self.output_path, "best.hdf5"),
                                    monitor="val_loss",
                                    verbose=1,
                                    save_weights_only = False,
                                    save_best_only=True,
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
        callbacks = [tensorboard, checkpoint, checkpoint_best]

        model.fit(train_dataset.make_one_shot_iterator(),
                  validation_data = validation_dataset.make_one_shot_iterator(),
                  callbacks=callbacks,
                  steps_per_epoch=num_train_batches,
                  validation_steps=num_validation_batches,
                  epochs=40)

        model.save(os.path.join(self.output_path, "model.h5"))

    def set_up_and_train(self):
        if self.model_path is None:
            model = self.set_up_model()
        else:
            model = load_model(self.model_path)
        self.train(model)
