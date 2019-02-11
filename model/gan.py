import datetime
import os
import sys
import time, datetime
import numpy as np
import tensorflow as tf
from skimage import io, color
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, Input, UpSampling2D, Conv2D, Conv1D, Dense, Dropout, BatchNormalization, Flatten, Conv2DTranspose, Reshape, Concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from DataGeneratorImages import DataGenerator
from tensorflow.python.client import device_lib
from utilities import mse_parse_function_gan, preprocess_and_return_X_batch
print(device_lib.list_local_devices())
image_size = 128

tf.logging.set_verbosity(tf.logging.ERROR)

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
		self.discriminator_input_shape = (128, 128, 2)

		def build_generator():
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
			return Model(inputs=generator_input, outputs=generator_output)

		def build_discriminator():
			discriminator_input = Input(self.discriminator_input_shape)
			#128
			discriminator_output = convLayer(discriminator_input, 32, (3, 3), stride=2, activation=None)
			discriminator_output = LeakyReLU(0.2)(discriminator_output)
			discriminator_output = BatchNormalization()(discriminator_output, training=False)
			#64
			discriminator_output = convLayer(discriminator_output, 16 , (3, 3), stride=2, activation=None)
			discriminator_output = LeakyReLU(0.2)(discriminator_output)
			discriminator_output = BatchNormalization()(discriminator_output, training=False)
			discriminator_output = Dropout(0.25)(discriminator_output)
			#32
			discriminator_output = convLayer(discriminator_output, 16 , (3, 3), stride=2, activation=None)
			discriminator_output = LeakyReLU(0.2)(discriminator_output)
			discriminator_output = BatchNormalization()(discriminator_output, training=False)
			discriminator_output = Dropout(0.25)(discriminator_output)
			#16
			discriminator_output = Flatten()(discriminator_output)
			discriminator_output = Dense(1, activation="sigmoid")(discriminator_output)

			return Model(inputs=discriminator_input, outputs=discriminator_output)

		self.generator = build_generator()
		self.generator.compile(loss='binary_crossentropy', optimizer = Adam(lr=.001))
		print("-----------------------Generator-----------------------")
		print(self.generator.summary())

		self.discriminator = build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy', optimizer = Adam(lr=.0001), metrics=['accuracy'])
		print("---------------------Discriminator---------------------")
		print(self.discriminator.summary())


		'''
		gan will take the L layer
		pass it through the generator
		pass the generated colorization through the discriminator
		output the discriminator score
		The target scores for these images will all be 1
		Since discriminator weights are locked, this means we will train the generator
		to produce the highest discriminator score possible
		'''
		gan_input = Input(shape = self.generator_input_shape)
		generated_colorization = self.generator(gan_input)
		# full_image = Concatenate(axis=3)([gan_input, generated_colorization])
		self.discriminator.trainable = False
		discriminator_judgement = self.discriminator(generated_colorization)

		gan = Model(inputs=gan_input, outputs=discriminator_judgement)
		gan.compile(loss='binary_crossentropy', optimizer = Adam(lr=.001))
		print("-------------------------GAN--------------------------")
		print(gan.summary())
		return gan

	def train(self, model):
		'''
		This function will carry out the training of the gan, including the discriminator step
		'''
		batch_size = 8
		self.output_path = self.output_path+"/"+datetime.datetime.now().strftime("%Y-%m-%d--%Hh%Mm")
		# os.mkdir(self.output_path)
		train_data_path = os.path.join(self.image_path, "Train_1")
		validation_data_path = os.path.join(self.image_path, "Validation_1")

		# partition = {"train": [], "validation": []}
		# for image in os.listdir(train_data_path):
		# 	partition["train"].append(os.path.join(train_data_path, image))

		# for image in os.listdir(validation_data_path):
		# 	partition["validation"].append(os.path.join(validation_data_path, image))

		# train_dataset = tf.data.Dataset.from_tensor_slices(partition["train"])
		# train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(len(partition["train"])))
		# train_dataset = train_dataset.map(mse_parse_function_gan, num_parallel_calls=8)
		# train_dataset = train_dataset.batch(batch_size)
		# train_dataset = train_dataset.prefetch(1)
		# train_iterator = train_dataset.make_one_shot_iterator()

		# validation_dataset = tf.data.Dataset.from_tensor_slices(partition["validation"])
		# validation_dataset = validation_dataset.shuffle(len(partition["validation"]))
		# validation_dataset = validation_dataset.map(mse_parse_function_gan, num_parallel_calls=4).repeat()
		# validation_dataset = validation_dataset.batch(batch_size)
		# validation_dataset = validation_dataset.prefetch(1)
		# validation_iterator = train_dataset.make_one_shot_iterator()
		
		datagen = image.ImageDataGenerator(rescale=(1./255))
		val_datagen = image.ImageDataGenerator(rescale=(1./255))
		epochs = 300
		num_train_batches = 258500//batch_size
		num_validation_batches = 10000//batch_size
		# num_train_batches = 32//batch_size
		# num_validation_batches = 32//batch_size
		real_images_labels = np.ones((batch_size, 1))
		generated_images_labels = np.zeros((batch_size, 1))
		start_time = time.time()
		def val_batch_generator(batch_size):
		    for val_batch in val_datagen.flow_from_directory(validation_data_path,
		                                             target_size=(image_size, image_size),
		                                             class_mode="input",
		                                             batch_size = batch_size):
				val_lab = color.rgb2lab(val_batch[0]).astype(np.float32)
				X = preprocess_and_return_X_batch(val_lab)
				Y = val_lab[:, :, :, 1:] / 128
				generated_ab_val = self.generator.predict(X, steps=1)
				disc_x_val = np.concatenate((Y, generated_ab_val), axis=0)
				disc_y_val = np.concatenate((real_images_labels, generated_images_labels), axis=0)
				yield ([disc_x_val, disc_y_val])

		for e in range(epochs):
			curr_batch = 0
			for batch in datagen.flow_from_directory(train_data_path,
		                                             target_size=(image_size, image_size),
		                                             class_mode="input",
		                                             batch_size=batch_size):
				lab = color.rgb2lab(batch[0]).astype(np.float32)
				X_train = preprocess_and_return_X_batch(lab)
				Y_train = lab[:, :, :, 1:] / 128
				generated_ab = self.generator.predict(X_train, steps=1)

				disc_x_train = np.concatenate((Y_train, generated_ab), axis=0)
				disc_y_train = np.concatenate((real_images_labels, generated_images_labels), axis=0)
				shuffle_indices = np.arange(disc_x_train.shape[0])
				np.random.shuffle(shuffle_indices)
				disc_x_train_shuffled = np.squeeze(disc_x_train[shuffle_indices])
				disc_y_train_shuffled = np.squeeze(disc_y_train[shuffle_indices])
				disc_loss, disc_acc = self.discriminator.train_on_batch(disc_x_train_shuffled, disc_y_train_shuffled)
				gen_loss = model.train_on_batch(X_train, real_images_labels)
				curr_batch = curr_batch + 1
				
				sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %f] [G loss: %f] Estimated time left: %s" % (e, epochs,
                                                                        curr_batch, num_train_batches,
                                                                        disc_loss, disc_acc,
                                                                        gen_loss,
                                                                        str(datetime.timedelta(seconds=((time.time() - start_time)/curr_batch) * (num_train_batches-curr_batch)))))
				sys.stdout.flush()
				if curr_batch >= num_train_batches:
					print(str(self.discriminator.metrics_names) + " : " + str(self.discriminator.evaluate_generator(val_batch_generator(batch_size), steps=num_validation_batches)))
					break
				#TODO: validation
				#TODO: record loss

	def set_up_and_train(self):
		if self.model_path is None:
			model = self.set_up_model()
		else:
			model = load_model(self.model_path)
		self.train(model)
