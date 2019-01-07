import datetime
import os, sys
import numpy as np
import tensorflow as tf
from skimage import io, color
import keras.backend as K
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Activation, Input, UpSampling2D, Conv2D, Conv1D, Dense, Dropout, BatchNormalization, Flatten, Conv2DTranspose, Reshape
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from model.DataGeneratorImages import DataGenerator
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

image_size = 64

# if tf.test.gpu_device_name():
#     print('Default GPU: {}'.format(tf.test.gpu_device_name()))
# else:
#     print('Failed to find default GPU.')
#     sys.exit(1)

# def categorical_crossentropy_color(y_pred, y_true):
#     closest_colors = K.argmax(y_true, axis=-1)
#     weights = K.gather(np.load("rebalance.npy").astype(np.float32), closest_colors)
#     weights = K.reshape(weights, (-1, 1))
#     #enhance rarer colors
#     y_true = y_true * weights

#     cross_entropy = K.categorical_crossentropy(y_pred, y_true)
#     cross_entropy = K.mean(cross_entropy, axis=-1)

#     return cross_entropy

def convLayer(input, filters, kernel_size, dilation=1, stride=1):
        return Conv2D(filters, kernel_size, padding="same", activation="relu", dilation_rate=dilation, strides=stride)(input)

class model:
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path
        print("Image path: " + image_path)
        print("Output path: " + output_path)

    # def prepare_test_data(self):
    #     test = []
    #     files = os.listdir(self.image_path + "/Test/Test")[:200]
    #     for image in files:
    #         img = read_image(image, self.image_path + "/Test/Test")
    #         if not img is None:
    #             img = np.array(img, dtype=np.float32)
    #             test.append(img)
    #     test = np.array(test, dtype=np.float32)

    #     test = test*(1.0/255)
    #     test = color.rgb2lab(test)
    #     test = preprocess(test)
    #     return test

    def set_up_model(self):
        #try keras sequential u dumdum
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
        model_output = Conv2D(313, (1, 1), padding="same")(model_output)
        model_output = Reshape((image_size*image_size, 313))(model_output)
        model_output = Activation("softmax")(model_output)
        return Model(inputs=model_input, outputs=model_output)

    def train(self, model):
        train_data_path = "Train_64"
        validation_data_path = "Validation"
        # os.system('gsutil -m cp -r ' + self.image_path + '/Train_Class_batches .')
        # os.system('gsutil -m cp -r ' + self.image_path + '/Train_small_batches .')
        

        partition = {"train": [], "validation": []}
        for image in os.listdir(train_data_path):
            partition["train"].append(os.path.join(train_data_path, image))
        
        for image in os.listdir(validation_data_path):
            partition["validation"].append(os.path.join(validation_data_path, image))
        
        batch_size = 64
        buckets = np.load("pts_in_hull.npy")
        rebalance = np.load("rebalance.npy")
        # num_train_batches = 4039
        num_train_batches = 2476
        num_validation_batches = 156
        params = {"dim": (image_size, image_size),
                  "batch_size": batch_size,
                  "shuffle": True}
        
        training_generator = DataGenerator(partition["train"], buckets, rebalance, **params)
        validation_generator = DataGenerator(partition["validation"], buckets, None, **params)
        model.summary()

        class WeightsSaver(Callback):
            def __init__(self, N, output_path):
                self.N = N
                self.output_path = output_path
                self.batch = 0

            def on_batch_end(self, batch, logs={}):
                if self.batch % self.N == 0:
                    name = 'currentWeights.h5'
                    self.model.save_weights(name)
                    try:
                        os.system('gsutil cp ' + name + ' ' + self.output_path)
                    except:
                        print("Could not upload current weights")
                self.batch += 1

        checkpoint = ModelCheckpoint("best.hdf5",
                                    monitor="accuracy",
                                    verbose=1,
                                    save_best_only=True,
                                    mode="max")

        every_2000_batches = WeightsSaver(1000, self.output_path)

        # every_10 = ModelCheckpoint("latest.hdf5",
        #                           monitor="accuracy",
        #                           verbose=1,
        #                           save_best_only=False,
        #                           mode='auto',
        #                           period=5)

        tensorboard = TensorBoard(log_dir=self.output_path, histogram_freq=0, write_images=True, update_freq=20000)
        callbacks = [tensorboard, checkpoint, every_2000_batches]
        # os.system('gsutil cp ' + self.image_path + '/Class_Train_R/currentWeights.h5 .')
        # model.load_weights("currentWeights.h5")
        model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=['accuracy'])

        model.fit_generator(training_generator, validation_data=validation_generator, validation_steps=num_validation_batches, callbacks=callbacks, steps_per_epoch=num_train_batches, epochs=7, workers=4, max_queue_size=8, use_multiprocessing=True) #5132 steps per epoch
        # model.fit_generator(batch_generator(batch_size), epochs=100, steps_per_epoch=5, validation_data=val_batch_generator(batch_size), validation_steps=5)

        # outputDate = now.strftime("%Y-%m-%d %Hh%Mm")
        # os.chdir("output")
        # os.mkdir(outputDate)
        # os.chdir(outputDate)
        try:
            model.save_weights("model_weights.h5")
            model.save("model.h5")
        # else:
        #     model.load_weights("/floyd/input/model/my_model_weights.h5")
        except:
            print("Could not save")

        os.system('gsutil cp model_weights.h5 ' + self.output_path)
        os.system('gsutil cp model.h5 ' + self.output_path)
        os.system('gsutil cp best.hdf5 ' + self.output_path)
        os.system('gsutil cp latest.hdf5 ' + self.output_path)

    def set_up_and_train(self):
        model = self.set_up_model()
        self.train(model)
