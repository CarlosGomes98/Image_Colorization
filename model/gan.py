import datetime
import os, sys
import numpy as np
import tensorflow as tf
from skimage import io, color
import tensorflow.keras.backend as K
from tensorflow.keras.optimiziers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, Input, UpSampling2D, Conv2D, Conv1D, Dense, Dropout, BatchNormalization, Flatten, Conv2DTranspose, Reshape, Concatenate
from tensorflow.keras.layers.advanced_activations import LeakyReLu
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from DataGeneratorImages import DataGenerator
from tensorflow.python.client import device_lib
from utilities import parse_function
print(device_lib.list_local_devices())

image_size = 128

def convLayer(input, filters, kernel_size, dilation=1, stride=1, activation="relu"):
    return Conv2D(filters, kernel_size, padding="same", activation=activation, dilation_rate=dilation, strides=stride)(input)

if tf.test.gpu_device_name():
    print('Default GPU: {}'.format(tf.test.gpu_device_name()))
else:
    print('Failed to find default GPU.')
    sys.exit(1)

class model:
    def __init__(self, image_path, output_path, model_path):
        self.image_path = image_path
		self.output_path = output_path
		self.model_path = model_path

    def set_up_model(self):
        '''
        Construct a network composed of the generator followed by the discriminator
        This follows the structure in which generator training is carried out:
            1. Get grayscale images on which to condition the generator
            2. Generate colourizations from those image_size
            3. Feed the images generated to the discriminator
            4. Use the output of the discriminator in order to train the generator
        After training, this is discarded and we use only the generator to create our colourization
        '''
        self.generator_input_shape = (128, 128, 1)
        self.discriminator_input_shape = (128, 128, 3)

        def build_generator(self):
            generator_input = Input(self.generator_input_shape)

            generator_output = convLayer(generator_input, 64, (3, 3))
    		generator_output = convLayer(generator_output, 64, (3, 3), stride=2)
    		generator_output = BatchNormalization()(generator_output)

    		generator_output = convLayer(generator_output, 128, (3, 3))
    		generator_output = convLayer(generator_output, 128, (3, 3), stride=2)
    		generator_output = BatchNormalization()(generator_output)

    		generator_output = convLayer(generator_output, 256, (3, 3))
    		generator_output = convLayer(generator_output, 256, (3, 3), stride=2)
    		generator_output = BatchNormalization()(generator_output)

    		generator_output = convLayer(generator_output, 512, (3, 3))
    		generator_output = convLayer(generator_output, 512, (3, 3))
    		generator_output = BatchNormalization()(generator_output)

    		generator_output = convLayer(generator_output, 512, (3, 3))
    		generator_output = convLayer(generator_output, 512, (3, 3))
    		generator_output = BatchNormalization()(generator_output)

    		generator_output = UpSampling2D((2, 2))(generator_output) #not sure if this or deconvolution
    		generator_output = convLayer(generator_output, 256, (3, 3))
    		generator_output = BatchNormalization()(generator_output)

    		generator_output = UpSampling2D((2, 2))(generator_output)
    		generator_output = convLayer(generator_output, 64, (3, 3))
    		generator_output = BatchNormalization()(generator_output)

    		generator_output = UpSampling2D((2, 2))(generator_output)
    		generator_output = Conv2D(2, (3, 3), activation="tanh", padding="same")(generator_output)
            #consider adding MSE term so output is close to input
    		return Model(inputs=generator_input, outputs=generator_output)

        def build_discriminator(self):
            discriminator_input = Input(self.discriminator_input_shape)
            #128
            discriminator_output = convLayer(discriminator_input, 32, (3, 3), stride=2, activation=None)
            discriminator_output = LeakyReLu(0.2)(discriminator_output)
            discriminator_output = BatchNormalization()(discriminator_output)
            #64
            discriminator_output = convLayer(discriminator_input, 16 , (3, 3), stride=2, activation=None)
            discriminator_output = LeakyReLu(0.2)(discriminator_output)
            discriminator_output = BatchNormalization()(discriminator_output)
            discriminator_output = Dropout(0.25)(discriminator_output)
            #32
            discriminator_output = convLayer(discriminator_input, 16 , (3, 3), stride=2, activation=None)
            discriminator_output = LeakyReLu(0.2)(discriminator_output)
            discriminator_output = BatchNormalization()(discriminator_output)
            discriminator_output = Dropout(0.25)(discriminator_output)
            #16
            discriminator_output = Flatten()(discriminator_output)
            discriminator_output = Dense(1, activation="sigmoid")(discriminator_output)

            return Model(inputs=discriminator_input, outputs=discriminator_output)

        self.generator = build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer = Adam(lr=.001))
        print("-----------------------Generator-----------------------")
        print(self.generator.summary())

        self.discriminator = build_discriminator();
        self.discriminator.compile(loss='binary_crossentropy', optimizer = Adam(lr=.0001))
        print("---------------------Discriminator---------------------")
        print(self.discriminator.summary())

        gan_input = Input(shape = self.generator_input_shape)
        #feed input to the generator
        generated_colorization = self.generator(gan_input)
        full_image = Concatenate([gan_input, generated_colorization])
        self.discriminator.trainable = False
        discriminator_judgement = self.discriminator(full_image)

        '''
        gan will take the L layer
        pass it through the generator
        pass the generated colorization through the discriminator
        output the discriminator score
        The target scores for these images will all be 1
        Since discriminator weights are locked, this means we will train the generator
        To produce the highest discriminator score possible
        '''
        gan = Model(inputs=gan_input, outputs=discriminator_judgement)
        gan.compile(loss='binary_crossentropy', optimizer = Adam(lr=.001))
        print("-------------------------GAN--------------------------")
        print(gan.summary())
