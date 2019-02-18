import datetime
import os
import sys
import time, datetime
import numpy as np
import tensorflow as tf
from skimage import io, color
import keras.backend as K
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Activation, Input, UpSampling2D, Conv2D, Conv1D, Dense, Dropout, BatchNormalization, Flatten, Conv2DTranspose, Reshape, Concatenate
from keras.layers import LeakyReLU
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
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

		# U-Net
		def build_generator():
			if self.model_path is not None:
				return load_model(os.path.join(self.model_path, "generator.h5"))
		
			generator_input = Input(self.generator_input_shape)
			# 128
			conv_1 = convLayer(generator_input, 64, (4, 4))
			conv_1 = BatchNormalization()(conv_1)
			# 64
			conv_2 = convLayer(conv_1, 64, (4, 4), stride=2)
			conv_2 = BatchNormalization()(conv_2)
			# 32
			conv_3 = convLayer(conv_2, 128, (4, 4), stride=2)
			conv_3 = BatchNormalization()(conv_3)
			# 16
			conv_4 = convLayer(conv_3, 256, (4, 4), stride=2)
			conv_4 = BatchNormalization()(conv_4)
			# 8
			conv_5 = convLayer(conv_4, 512, (4, 4), stride=2)
			conv_5 = BatchNormalization()(conv_5)
			# 4
			conv_6 = convLayer(conv_5, 512, (4, 4), stride=2)
			conv_6 = BatchNormalization()(conv_6)
			# 2
			conv_7 = convLayer(conv_6, 512, (4, 4), stride=2)
			conv_7 = BatchNormalization()(conv_7)
			# 4
			conv_8 = UpSampling2D((2, 2))(conv_7)
			conv_8 = convLayer(conv_8, 512, (4, 4))
			conv_8 = BatchNormalization()(conv_8)
			conv_8 = Concatenate(axis=-1)([conv_6, conv_8])
			# 8
			conv_9 = UpSampling2D((2, 2))(conv_8)
			conv_9 = convLayer(conv_9, 512, (4, 4))
			conv_9 = BatchNormalization()(conv_9)
			conv_9 = Concatenate(axis=-1)([conv_5, conv_9])
			# 16
			conv_10 = UpSampling2D((2, 2))(conv_9)
			conv_10 = convLayer(conv_10, 512, (4, 4))
			conv_10 = BatchNormalization()(conv_10)
			conv_10 = Concatenate(axis=-1)([conv_4, conv_10])
			# conv_10 = Dropout(0.25)(conv_10)
			# 32
			conv_11 = UpSampling2D((2, 2))(conv_10)
			conv_11 = convLayer(conv_11, 256, (4, 4))
			conv_11 = BatchNormalization()(conv_11)
			conv_11 = Concatenate(axis=-1)([conv_3, conv_11])
			# conv_11 = Dropout(0.25)(conv_11)
			# 64
			conv_12 = UpSampling2D((2, 2))(conv_11)
			conv_12 = convLayer(conv_12, 128, (4, 4))
			conv_12 = BatchNormalization()(conv_12)
			conv_12 = Concatenate(axis=-1)([conv_2, conv_12])
			# conv_12 = Dropout(0.25)(conv_12)
			# 128
			conv_13 = UpSampling2D((2, 2))(conv_12)
			conv_13 = convLayer(conv_13, 64, (4, 4))
			conv_13 = BatchNormalization()(conv_13)
			conv_13 = Concatenate(axis=-1)([conv_1, conv_13])

			generator_output = Conv2D(2, (1, 1), activation="tanh", padding="same")(conv_13)
			
			generator = Model(inputs=generator_input, outputs=generator_output)
			generator.compile(loss='binary_crossentropy', optimizer = Adam(lr=.0002, beta_1 = 0.5))
			return generator

		def build_discriminator():
			if self.model_path is not None:
				return load_model(os.path.join(self.model_path, "discriminator.h5"))
		
			discriminator_input = Input(self.discriminator_input_shape)
			#128
			discriminator_output = convLayer(discriminator_input, 32, (3, 3), stride=2, activation=None)
			discriminator_output = LeakyReLU(0.2)(discriminator_output)
			discriminator_output = BatchNormalization()(discriminator_output)
			#64
			discriminator_output = convLayer(discriminator_output, 16 , (3, 3), stride=2, activation=None)
			discriminator_output = LeakyReLU(0.2)(discriminator_output)
			discriminator_output = BatchNormalization()(discriminator_output)
			discriminator_output = Dropout(0.25)(discriminator_output)
			#32
			discriminator_output = convLayer(discriminator_output, 16 , (3, 3), stride=2, activation=None)
			discriminator_output = LeakyReLU(0.2)(discriminator_output)
			discriminator_output = BatchNormalization()(discriminator_output)
			discriminator_output = Dropout(0.25)(discriminator_output)
			#16
			discriminator_output = Flatten()(discriminator_output)
			discriminator_output = Dense(1, activation="sigmoid")(discriminator_output)

			discriminator = Model(inputs=discriminator_input, outputs=discriminator_output)
			discriminator.compile(loss='binary_crossentropy', optimizer = Adam(lr=.001, beta_1 = 0.5), metrics=['accuracy'])
			return discriminator

		self.generator = build_generator()
		#maybe add another term to the loss, mse
		print("-----------------------Generator-----------------------")
		print(self.generator.summary())

		self.discriminator = build_discriminator()
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
		gan.compile(loss='binary_crossentropy', optimizer = Adam(lr=.001, beta_1 = 0.5))
		print("-------------------------GAN--------------------------")
		print(gan.summary())
		return gan

	def train(self, model):
		'''
		This function will carry out the training of the gan, including the discriminator step
		'''
		batch_size = 16
		self.output_path = os.path.join(self.output_path, "full_u_net")#datetime.datetime.now().strftime("%Y-%m-%d--%Hh%Mm"))
		os.mkdir(self.output_path)
		os.mkdir(os.path.join(self.output_path, "images"))
		self.writer = tf.summary.FileWriter(self.output_path)
		train_data_path = os.path.join(self.image_path, "Train_small")
		validation_data_path = os.path.join(self.image_path, "Train_small")

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
		epochs = 1000
		disc_training_steps = 1
		# num_train_batches = 258500//batch_size
		# num_validation_batches = 10000//batch_size
		num_train_batches = 32//batch_size
		num_validation_batches = 32//batch_size

		#one sided smoothing (https://arxiv.org/pdf/1606.03498.pdf)
		real_images_labels = np.full((batch_size, 1), 0.9)
		generated_images_labels = np.full((batch_size, 1), 0.1)
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

				# disc_x_train = np.concatenate((Y_train, generated_ab), axis=0)
				# disc_y_train = np.concatenate((real_images_labels, generated_images_labels), axis=0)
				
				# shuffle_indices = np.arange(disc_x_train.shape[0])
				# np.random.shuffle(shuffle_indices)
				
				# disc_x_train_shuffled = np.squeeze(disc_x_train[shuffle_indices])
				# disc_y_train_shuffled = np.squeeze(disc_y_train[shuffle_indices])
				
				#train on real batch then on fake batch (https://github.com/soumith/ganhacks/blob/master/README.md point 4)
				#Set learning phase manually due to Dropout and BatchNormalization layers
				K.set_learning_phase(1)
				#TODO: maybe train whenever acc falls below a certain threshold (90?)
				for i in range(disc_training_steps):
					disc_loss_r, disc_acc_r = self.discriminator.train_on_batch(Y_train, real_images_labels)
					disc_loss_f, disc_acc_f = self.discriminator.train_on_batch(generated_ab, generated_images_labels)
				
				disc_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="disc_loss", simple_value=(disc_loss_r + disc_loss_f) / 2)])
				# disc_acc_summary = tf.Summary(value=[tf.Summary.Value(tag="disc_acc", simple_value=(disc_acc_r + disc_acc_f) / 2)])
				
				gen_loss = model.train_on_batch(X_train, real_images_labels)
				gen_loss_summary = tf.Summary(value=[tf.Summary.Value(tag="gen_loss", simple_value=gen_loss)])
				
				K.set_learning_phase(0)

				current_step = (e) * num_train_batches + curr_batch
				self.writer.add_summary(disc_loss_summary, current_step*disc_training_steps)
				# self.writer.add_summary(disc_acc_summary, current_step*disc_training_steps)
				self.writer.add_summary(gen_loss_summary, current_step)
				curr_batch = curr_batch + 1
				
				sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] Estimated time left: %s" % (e, epochs,
                                                                        curr_batch, num_train_batches,
                                                                        (disc_loss_r + disc_loss_f)/2,
                                                                        gen_loss,
                                                                        str(datetime.timedelta(seconds=((time.time() - start_time)/curr_batch) * (num_train_batches-curr_batch)))))
				sys.stdout.flush()
				if curr_batch >= num_train_batches:
					if e % 250 == 0:
						print(str(self.discriminator.metrics_names) + " : " + str(self.discriminator.evaluate_generator(val_batch_generator(batch_size), steps=num_validation_batches)))
						model.save(os.path.join(self.output_path, "model.h5"))
						self.discriminator.save(os.path.join(self.output_path, "discriminator.h5"))
						self.generator.save(os.path.join(self.output_path, "generator.h5"))

						images = np.concatenate((X_train * 50 + 50, generated_ab*128), axis=3)
						os.mkdir(os.path.join(self.output_path, "images", "epoch_{}".format(e)))
						for i in range(batch_size):
							curr_image = color.lab2rgb(images[i])
							io.imsave(os.path.join(self.output_path, "images", "epoch_{}".format(e), "image_{}.jpg".format(i)), curr_image)
					break

	def set_up_and_train(self):
		model = self.set_up_model()
		self.train(model)
