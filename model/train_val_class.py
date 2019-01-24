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
from utilities import parse_function
print(device_lib.list_local_devices())

image_size = 128

if tf.test.gpu_device_name():
    print('Default GPU: {}'.format(tf.test.gpu_device_name()))
else:
    print('Failed to find default GPU.')
    sys.exit(1)

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
        self.current_epoch = 0
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
        return Model(inputs=model_input, outputs=model_output)

    def train(self, model):
        self.output_path = self.output_path+"/"+datetime.datetime.now().strftime("%Y-%m-%d %Hh%Mm")
        os.mkdir(self.output_path)
        train_data_path = os.path.join(self.image_path, "Train")
        validation_data_path = os.path.join(self.image_path, "Validation")
        # os.system('gsutil -m cp -r ' + self.image_path + '/Train_Class_batches .')
        # os.system('gsutil -m cp -r ' + self.image_path + '/Train_small_batches .')
        
        batch_size = 16

        partition = {"train": [], "validation": []}
        for image in os.listdir(train_data_path):
            partition["train"].append(os.path.join(train_data_path, image))
        
        for image in os.listdir(validation_data_path):
            partition["validation"].append(os.path.join(validation_data_path, image))
        
        train_dataset = tf.data.Dataset.from_tensor_slices(partition["train"]).repeat()
        train_dataset = train_dataset.shuffle(len(partition["train"]))
        train_dataset = train_dataset.map(parse_function, num_parallel_calls=8)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(1)

        validation_dataset = tf.data.Dataset.from_tensor_slices(partition["validation"]).repeat()
        validation_dataset = validation_dataset.shuffle(len(partition["validation"]))
        validation_dataset = validation_dataset.map(parse_function, num_parallel_calls=1)
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

        checkpoint_best = ModelCheckpoint(os.path.join(self.output_path, "epoch_"+str(self.get_epoch())+".hdf5"),
                                    monitor="val_acc",
                                    verbose=1,
                                    save_best_only=False,
                                    mode="auto")

        every_epoch = WeightsSaver(num_train_batches, self.output_path)

        # every_10 = ModelCheckpoint("latest.hdf5",
        #                           monitor="accuracy",
        #                           verbose=1,
        #                           save_best_only=False,
        #                           mode='auto',
        #                           period=5)

        tensorboard = TensorBoard(log_dir=self.output_path, histogram_freq=0, write_images=True)
        callbacks = [tensorboard, checkpoint_best]
        # os.system('gsutil cp ' + self.image_path + '/Class_64_64_zip/currentWeights.h5 .')
        # model = load_model("currentWeights.h5")
        model.compile(loss="categorical_crossentropy",
                    optimizer="adam",
                    metrics=['accuracy'])

        model.fit(train_dataset.make_one_shot_iterator(),
                  validation_data = validation_dataset.make_one_shot_iterator(),
                  callbacks=callbacks,
                  steps_per_epoch=num_train_batches,
                  validation_steps=num_validation_batches,
                  epochs=5)

        model.save(os.path.join(self.output_path, "model.h5"))
        # outputDate = now.strftime("%Y-%m-%d %Hh%Mm")
        # os.chdir("output")
        # os.mkdir(outputDate)
        # os.chdir(outputDate)
        # try:
        #     model.save_weights("last_model_weights.h5")
        #     model.save("model.h5")
        # # else:
        # #     model.load_weights("/floyd/input/model/my_model_weights.h5")
        # except:
        #     print("Could not save")

        # os.system('gsutil cp last_model_weights.h5 ' + self.output_path)
        # os.system('gsutil cp model.h5 ' + self.output_path)
        # os.system('gsutil cp best.hdf5 ' + self.output_path)
        # os.system('gsutil cp latest.hdf5 ' + self.output_path)

    def set_up_and_train(self):
        model = self.set_up_model()
        self.train(model)
