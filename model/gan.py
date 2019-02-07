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
        return gan

    def train(self, model):
        '''
        This function will carry out the training of the gan, including the discriminator step
        '''
        batch_size = 16
        self.output_path = self.output_path+"/"+datetime.datetime.now().strftime("%Y-%m-%d--%Hh%Mm")
        os.mkdir(self.output_path)
        train_data_path = os.path.join(self.image_path, "Train")
        validation_data_path = os.path.join(self.image_path, "Validation")

        partition = {"train": [], "validation": []}
        for image in os.listdir(train_data_path):
            partition["train"].append(os.path.join(train_data_path, image))

        for image in os.listdir(validation_data_path):
            partition["validation"].append(os.path.join(validation_data_path, image))

        train_dataset = tf.data.Dataset.from_tensor_slices(partition["train"])
        train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(len(partition["train"])))
        train_dataset = train_dataset.map(mse_parse_function, num_parallel_calls=8)
        train_dataset = train_dataset.batch(batch_size/2)
        train_dataset = train_dataset.prefetch(1)
        train_iterator = train_dataset.make_one_shot_iterator()

        validation_dataset = tf.data.Dataset.from_tensor_slices(partition["validation"])
        validation_dataset = validation_dataset.shuffle(len(partition["validation"]))
        validation_dataset = validation_dataset.map(mse_parse_function, num_parallel_calls=4).repeat()
        validation_dataset = validation_dataset.batch(batch_size/2)
        validation_dataset = validation_dataset.prefetch(1)
        validation_iterator = train_dataset.make_one_shot_iterator()

        epochs = 5
        num_train_batches = 258500//batch_size
        num_validation_batches = 10000//batch_size
        for e in range(epochs):
            for b in range(num_train_batches):
                X_train, Y_train = train_iterator.next()
                X_val, Y_val = validation_iterator.next()

                generated_ab = self.generator.predict(X_train)

                real_images = np.concatenate(X_train, Y_train, axis=2)
                real_images_labels = np.ones((batch_size/2, 1))
                generated_images = np.concatenate(X_train, generated_ab, axis=2)
                generated_images_labels = np.zeros((batch_size/2, 1))

                disc_x_train = np.concatenate(real_images, generated_images)
                disc_y_train = np.concatenate(real_images_labels, generated_images_labels)

                generated_ab_val = self.generator.predict(X_val)

                real_images_val = np.concatenate(X_val, Y_val, axis=2)
                real_images_val_labels = np.ones((batch_size/2, 1))
                generated_images_val = np.concatenate(X_val, generated_ab, axis=2)
                generated_images_val_labels = np.zeros((batch_size/2, 1))

                disc_x_val = np.concatenate(real_images_val, generated_images_val)
                disc_y_val = np.concatenate(real_images_val_labels, generated_images_val_labels)

                self.discriminator.train_on_batch(disc_x_train, disc_y_train)

                #TODO: validation
                #TODO: record loss

    def set_up_and_train(self):
        if self.model_path is None:
            model = self.set_up_model()
        else:
            model = load_model(self.model_path)
        self.train(model)
