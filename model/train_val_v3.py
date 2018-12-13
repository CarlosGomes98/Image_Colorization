import datetime
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input, UpSampling2D, Conv2D, Dense, Dropout, BatchNormalization, Flatten, Conv2DTranspose
import os, sys
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
from skimage import io, color
from keras.preprocessing import image
import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from utilities.utilities import read_image, show_image, preprocess_and_return_X, convLayer
image_size = 256

class model:
	image_path=""
	output_path=""

	def __init__(self, image_path, output_path):
		self.image_path = image_path
		self.output_path = output_path

	def set_up_model(self):
		input_shape = (image_size, image_size, 1)

		model_input = Input(shape = input_shape)

		model_output = convLayer(model_input, 64, (3, 3))
		model_output = convLayer(model_output, 64, (3, 3), stride=2)
		model_output = BatchNormalization()(model_output)

		model_output = convLayer(model_output, 128, (3, 3))
		model_output = convLayer(model_output, 128, (3, 3), stride=2)
		model_output = BatchNormalization()(model_output)

		model_output = convLayer(model_output, 256, (3, 3))
		model_output = convLayer(model_output, 256, (3, 3), stride=2)
		model_output = BatchNormalization()(model_output)

		model_output = convLayer(model_output, 512, (3, 3))
		model_output = convLayer(model_output, 512, (3, 3))
		model_output = BatchNormalization()(model_output)

		model_output = convLayer(model_output, 512, (3, 3))
		model_output = convLayer(model_output, 512, (3, 3))
		model_output = BatchNormalization()(model_output)

		model_output = UpSampling2D((2, 2))(model_output) #not sure if this or deconvolution
		model_output = convLayer(model_output, 256, (3, 3))
		model_output = BatchNormalization()(model_output)

		model_output = UpSampling2D((2, 2))(model_output)
		model_output = convLayer(model_output, 64, (3, 3))
		model_output = BatchNormalization()(model_output)

		model_output = UpSampling2D((2, 2))(model_output)
		model_output = Conv2D(2, (3, 3), activation="tanh", padding="same")(model_output)

		return Model(inputs=model_input, outputs=model_output)

	def train(self, model):
		datagen = ImageDataGenerator(rescale=(1./255))

		val_datagen = ImageDataGenerator(rescale=(1./255))
        # Download Train and Validation data
		# os.system('gsutil -m cp -r ' + self.image_path + '/Train .')
		os.system('gsutil -m cp -r ' + self.image_path + '/Validation .')

		batch_size = 32
		def batch_generator(batch_size):
		    for batch in datagen.flow_from_directory("Validation",
		                                             target_size=(image_size, image_size),
		                                             class_mode="input",
		                                             batch_size = batch_size):
		        lab = color.rgb2lab(batch[0])
		        X = preprocess_and_return_X(lab)
		        Y = lab[:, :, :, 1:] / 128
		        yield ([X, Y])

		def val_batch_generator(batch_size):
		    for batch in val_datagen.flow_from_directory("Validation",
		                                             target_size=(image_size, image_size),
		                                             class_mode="input",
		                                             batch_size = batch_size):
		        lab = color.rgb2lab(batch[0])
		        X = preprocess_and_return_X(lab)
		        Y = lab[:, :, :, 1:] / 128
		        yield ([X, Y])

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

		every_20_batches = WeightsSaver(20, self.output_path)


		tensorboard = TensorBoard(log_dir=".")
		callbacks = [tensorboard, checkpoint, every_20_batches]

        # Uncomment to continue previous training
        # model.load_weights("/content/drive/My Drive/app/output/2018-09-29 14:58/latest.hdf5")#manually change this, i know, i know

		model.compile(loss='mean_squared_error',
		            optimizer="adam",
		            metrics=['accuracy'])

		model.fit_generator(batch_generator(batch_size), callbacks=callbacks, epochs=3, steps_per_epoch=4040, validation_data=val_batch_generator(batch_size), validation_steps=157) #5132 steps per epoch

		try:
			model.save_weights("model_weights.h5")
			model.save("model.h5")
		except:
			print("Could not save")

		os.system('gsutil cp model_weights.h5 ' + self.output_path)
		os.system('gsutil cp model.h5 ' + self.output_path)
		os.system('gsutil cp best.hdf5 ' + self.output_path)
		os.system('gsutil cp latest.h5 ' + self.output_path)

	# # Test model
	# def test(self, test, model):
	# 	output = model.predict(test)
	# 	output = output * 128
	# 	# Output colorizations
	# 	for i in range(len(output)):
	# 	    cur = np.zeros((image_size, image_size, 3))
	# 	    cur[:,:,0] = test[i][:,:,0]
	# 	    cur[:,:,1:] = output[i]
	# 	    io.imsave(self.output_path + "/" + str(i) + ".png", color.lab2rgb(cur))

	def set_up_and_train(self):
		model = self.set_up_model()
		self.train(model)
