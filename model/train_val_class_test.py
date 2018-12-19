import datetime
import os, sys
import numpy as np
import tensorflow as tf
from skimage import io, color
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D, Conv2D, Conv1D, Dense, Dropout, BatchNormalization, Flatten, Conv2DTranspose, Reshape
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from utilities import read_image, show_image, preprocess_and_return_X, convLayer, bucketize_gaussian, decode_bucketize_images
from DataGenerator import DataGenerator
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

image_size = 128

class model:
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path

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
        input_shape = (image_size, image_size, 1)

        model_input = Input(shape = input_shape)

        # conv1
        model_output = convLayer(model_input, 64, (3, 3))
        model_output = convLayer(model_output, 64, (3, 3), stride=2)
        model_output = BatchNormalization()(model_output)
        # conv2
        # model_output = convLayer(model_output, 128, (3, 3))
        model_output = convLayer(model_output, 128, (3, 3), stride=2)
        model_output = BatchNormalization()(model_output)
        # conv3
        # model_output = convLayer(model_output, 256, (3, 3))
        # model_output = convLayer(model_output, 256, (3, 3))
        model_output = convLayer(model_output, 256, (3, 3), stride=2)
        model_output = BatchNormalization()(model_output)
        # conv4
        # model_output = convLayer(model_output, 512, (3, 3))
        model_output = convLayer(model_output, 512, (3, 3))
        model_output = convLayer(model_output, 512, (3, 3))
        model_output = BatchNormalization()(model_output)
        # conv5
        # model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        # model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        # model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        # model_output = BatchNormalization()(model_output)
        # conv6
        # model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        # model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        # model_output = convLayer(model_output, 512, (3, 3), dilation=2)
        # model_output = BatchNormalization()(model_output)
        # conv7
        # model_output = convLayer(model_output, 256, (3, 3))
        # model_output = convLayer(model_output, 256, (3, 3))
        # model_output = convLayer(model_output, 256, (3, 3))
        # model_output = BatchNormalization()(model_output)
        # conv8
        model_output = UpSampling2D((2, 2))(model_output)
        model_output = convLayer(model_output, 256, (3, 3))
        model_output = UpSampling2D((2, 2))(model_output)
        model_output = convLayer(model_output, 256, (3, 3))
        model_output = UpSampling2D((2, 2))(model_output)
        model_output = convLayer(model_output, 256, (3, 3))

        # unary prediction
        reshaped_output = Reshape((image_size*image_size, 256))(model_output)
        model_output = Conv1D(313, (1), padding="same", activation="softmax")(reshaped_output)
        return Model(inputs=model_input, outputs=model_output)

    def train(self, model):
        # os.system('gsutil -m cp -r ' + self.image_path + '/Train .')
        # os.system('gsutil -m cp -r ' + self.image_path + '/Validation .')
        # os.system('gsutil -m cp -r ' + self.image_path + '/pts_in_hull.npy .')
        # os.system('gsutil -m cp -r ' + self.image_path + '/prior_probs.npy .')


        #download rebalance factors and quantization files
        batch_size = 16
        rebalance = np.load("model/rebalance.npy")
        buckets = np.load("model/pts_in_hull.npy")
        
        
        partition = {"train": [], "validation": []}
        train_data_path = self.image_path + "/Train_small/Train_small"
        validation_data_path = self.image_path + "/Train_small/Train_small"
        for image in os.listdir(self.image_path + "/Train_small/Train_small"):
            partition["train"].append(image)
        
        for image in os.listdir(self.image_path + "/Train_small/Train_small"):
            partition["validation"].append(image)
        
        labels = {}
        for index, image in enumerate(partition["train"]):
            labels[image] = index    
        for index, image in enumerate(partition["validation"]):
            labels[image] = index

        params = {"dim": (image_size, image_size),
                  "batch_size": batch_size,
                  "n_channels": 313,
                  "shuffle": True}
        
        quant = (buckets, rebalance)
        training_generator = DataGenerator(partition["train"], labels, train_data_path, *quant, **params)
        validation_generator = DataGenerator(partition["validation"], labels, validation_data_path, *quant, **params)

        class WeightsSaver(Callback):
            def __init__(self, N, output_path):
                self.N = N
                self.output_path = output_path
                self.batch = 0

            def on_batch_end(self, batch, logs={}):
                if self.batch % self.N == 0:
                    name = 'currentWeights.h5'
                    self.model.save_weights(name)
                    # try:
                    #     os.system('gsutil cp ' + name + ' ' + self.output_path)
                    # except:
                    #     print("Could not upload current weights")
                self.batch += 1

        checkpoint = ModelCheckpoint("best.hdf5",
                                    monitor="accuracy",
                                    verbose=1,
                                    save_best_only=True,
                                    mode="max")

        every_20_batches = WeightsSaver(20, self.output_path)

        every_10 = ModelCheckpoint("latest.hdf5",
                                  monitor="accuracy",
                                  verbose=1,
                                  save_best_only=False,
                                  mode='auto',
                                  period=1)

        tensorboard = TensorBoard(log_dir=".")
        callbacks = [tensorboard, checkpoint, every_10, every_20_batches]
        # model.load_weights("D:/Libraries/Documents/Image_Colorization/output/model_weights.h5")

        model.compile(loss='categorical_crossentropy',
                    optimizer="adam",
                    metrics=['accuracy'])

        model.fit_generator(training_generator, epochs=100, steps_per_epoch=10, workers=4, max_queue_size=8, use_multiprocessing=False) #5132 steps per epoch

        # outputDate = now.strftime("%Y-%m-%d %Hh%Mm")
        # os.chdir("output")
        # os.mkdir(outputDate)
        # os.chdir(outputDate)
        try:
            model.save_weights(self.output_path + "/model_weights.h5")
            model.save(self.output_path + "/model.h5")
        # else:
        #     model.load_weights("/floyd/input/model/my_model_weights.h5")
        except:
            print("Could not save")

        # os.system('gsutil cp model_weights.h5 ' + self.output_path)
        # os.system('gsutil cp model.h5 ' + self.output_path)
        # os.system('gsutil cp best.hdf5 ' + self.output_path)
        # os.system('gsutil cp latest.h5 ' + self.output_path)


    # In[ ]:

    '''
    # model.load_weights("drive/app/output/my_model_weights2018-08-10 20:36.h5")
    # if(not os.path.exists("/floyd/input/model/my_model_weights.h5")):
    datagen = ImageDataGenerator(
        #featurewise_center=True,
        #samplewise_center=False,
        shear_range=0.4,
        zoom_range=0.4,
        rotation_range=40,
        horizontal_flip=True,
        rescale=(1./255)
    )


    batch_size = 64
    def batch_generator(batch_size):
        for batch in datagen.flow_from_directory(self.image_path+"/Train/",
                                                 target_size=(image_size, image_size),
                                                 class_mode="input",
                                                 batch_size = batch_size):
            lab = color.rgb2lab(batch[0])
            X = preprocess(lab)
            Y = lab[:, :, :, 1:] / 128
            yield ([X, Y])

    def val_batch_generator(batch_size):
        for batch in datagen.flow_from_directory(self.image_path+"/Validation/",
                                             target_size=(image_size, image_size),
                                             class_mode="input",
                                             batch_size = batch_size):
            lab = color.rgb2lab(batch[0])
            X = preprocess(lab)
            Y = lab[:, :, :, 1:] / 128
            yield ([X, Y])

    model.summary()

    outputDate = now.strftime("%Y-%m-%d %H:%M")
    os.chdir("output")
    os.mkdir(outputDate)
    os.chdir(outputDate)

    class WeightsSaver(Callback):
        def __init__(self, N):
            self.N = N
            self.batch = 0

        def on_batch_end(self, batch, logs={}):
            if self.batch % self.N == 0:
                name = 'currentWeights.h5'
                self.model.save_weights(name)
            self.batch += 1

    checkpoint = ModelCheckpoint("best.hdf5",
                                monitor="accuracy",
                                verbose=1,
                                save_best_only=True,
                                mode="max")

    every_20_batches = WeightsSaver(20)

    every_10 = ModelCheckpoint("latest.hdf5",
                              monitor="accuracy",
                              verbose=1,
                              save_best_only=False,
                              mode='auto',
                              period=1)

    tensorboard = TensorBoard(log_dir=".")


    model.load_weights("/content/drive/My Drive/app/output/2018-09-29 14:58/latest.hdf5")#manually change this, i know, i know

    model.compile(loss='mean_squared_error',
                optimizer="adam",
                metrics=['accuracy'])

    model.fit_generator(batch_generator(batch_size), callbacks=[tensorboard, checkpoint, every_10, every_20_batches], epochs=1, steps_per_epoch=5132, validation_data=val_batch_generator(batch_size)) #5132 steps per epoch


    try:
        model.save_weights("model_weights.h5")
    # else:
    #     model.load_weights("/floyd/input/model/my_model_weights.h5")
    except:
        print("Could not save")

    model.save("model.h5")


    # In[ ]:


    model.load_weights("latest.hdf5")

    model.compile(loss='mean_squared_error',
                optimizer="adam",
                metrics=['accuracy'])

    '''
    # In[ ]:


    # # Test model
    # def test(self, test, model):
    #     output = model.predict(test)
    #     output = output * 128
    #     # Output colorizations
    #     for i in range(len(output)):
    #         cur = np.zeros((image_size, image_size, 3))
    #         cur[:,:,0] = test[i][:,:,0]
    #         cur[:,:,1:] = output[i]
    #         io.imsave(self.output_path + "/" + str(i) + ".png", color.lab2rgb(cur))

    def set_up_and_train(self):
        model = self.set_up_model()
        self.train(model)
