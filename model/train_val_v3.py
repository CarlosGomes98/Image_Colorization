import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D, Dense, Dropout, BatchNormalization, Flatten, Conv2DTranspose
import os, sys
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import numpy as np
from skimage import io, color
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
from utilities import preprocess_and_return_X_batch, mse_parse_function
image_size = 256

def convLayer(input, filters, kernel_size, dilation=1, stride=1):
    return Conv2D(filters, kernel_size, padding="same", activation="relu", dilation_rate=dilation, strides=stride)(input)

class model:
	def __init__(self, image_path, output_path, model_path):
		self.image_path = image_path
		self.output_path = output_path
		self.model_path = model_path

	def set_up_model(self):
		input_shape = (image_size, image_size, 1)

		model_input = Input(shape = input_shape)
		# 256 => 128
		model_output = convLayer(model_input, 64, (3, 3))
		model_output = convLayer(model_output, 64, (3, 3), stride=2)
		model_output = BatchNormalization()(model_output)
		# 128 => 64
		model_output = convLayer(model_output, 128, (3, 3))
		model_output = convLayer(model_output, 128, (3, 3), stride=2)
		model_output = BatchNormalization()(model_output)
		# 64 => 32
		model_output = convLayer(model_output, 256, (3, 3))
		model_output = convLayer(model_output, 256, (3, 3), stride=2)
		model_output = BatchNormalization()(model_output)

		model_output = convLayer(model_output, 512, (3, 3))
		model_output = convLayer(model_output, 512, (3, 3))
		model_output = BatchNormalization()(model_output)

		model_output = convLayer(model_output, 512, (3, 3))
		model_output = convLayer(model_output, 512, (3, 3))
		model_output = BatchNormalization()(model_output)
		# 32 => 64
		model_output = UpSampling2D((2, 2))(model_output)
		model_output = convLayer(model_output, 256, (3, 3))
		model_output = BatchNormalization()(model_output)
		# 64 => 128
		model_output = UpSampling2D((2, 2))(model_output)
		model_output = convLayer(model_output, 128, (3, 3))
		model_output = BatchNormalization()(model_output)
		# 128 => 256
		model_output = UpSampling2D((2, 2))(model_output)
		model_output = convLayer(model_output, 64, (3, 3))
		model_output = Conv2D(2, (1, 1), activation="tanh", padding="same")(model_output)

		model = Model(inputs=model_input, outputs=model_output)
		
		model.compile(loss='mean_squared_error',
		            optimizer="adam",
		            metrics=['accuracy'])

		return model

	def train(self, model):
		self.output_path = self.output_path+"/flowers"#+datetime.datetime.now().strftime("%Y-%m-%d--%Hh%Mm")
		os.mkdir(self.output_path)
		train_data_path = os.path.join(self.image_path, "flowers", "flowers")
		validation_data_path = os.path.join(self.image_path, "flowers_val", "flowers_val")
        # Download Train and Validation data
		# os.system('gsutil -m cp -r ' + self.image_path + '/Train .')
		# os.system('gsutil -m cp -r ' + self.image_path + '/Validation .')

		batch_size = 16
		# num_train_batches = 258500//batch_size
		# num_validation_batches = 10000//batch_size
		num_train_batches = 7189//batch_size
		num_validation_batches = 1000//batch_size
		partition = {"train": [], "validation": []}
		for image in os.listdir(train_data_path):
			partition["train"].append(os.path.join(train_data_path, image))
        
		for image in os.listdir(validation_data_path):
			partition["validation"].append(os.path.join(validation_data_path, image))
        
		train_dataset = tf.data.Dataset.from_tensor_slices(partition["train"])
		train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(len(partition["train"])))
		train_dataset = train_dataset.map(mse_parse_function, num_parallel_calls=8)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(1)

		validation_dataset = tf.data.Dataset.from_tensor_slices(partition["validation"])
		validation_dataset = validation_dataset.shuffle(len(partition["validation"]))
		validation_dataset = validation_dataset.map(mse_parse_function, num_parallel_calls=4).repeat()
		validation_dataset = validation_dataset.batch(batch_size)
		validation_dataset = validation_dataset.prefetch(1)

		model.summary()

		class WeightsSaver(Callback):
			def __init__(self, N, output_path):
				self.N = N
				self.output_path = output_path
				self.batch = 0

			def on_batch_end(self, batch, logs={}):
				if self.batch % self.N == 0:
					name = 'currentWeights.h5'
					self.model.save(os.path.join(self.output_path, "model.h5"))
				self.batch += 1

		checkpoint = ModelCheckpoint(os.path.join(self.output_path, "curr.hdf5"),
                                    monitor="val_loss",
                                    verbose=1,
                                    save_weights_only = False,
                                    save_best_only=False,
                                    mode="auto",
                                    period=1)
		
		best_checkpoint = ModelCheckpoint(os.path.join(self.output_path, "best.hdf5"),
                                    monitor="val_loss",
                                    verbose=1,
                                    save_weights_only = False,
                                    save_best_only=True,
                                    mode="auto",
                                    period=1)


		every_20_batches = WeightsSaver(20, self.output_path)


		tensorboard = TensorBoard(log_dir=self.output_path, histogram_freq=0, write_images=True)
		callbacks = [tensorboard, checkpoint, best_checkpoint]

        # Uncomment to continue previous training
        # model.load_weights("/content/drive/My Drive/app/output/2018-09-29 14:58/latest.hdf5")#manually change this, i know, i know

		model.fit(train_dataset.make_one_shot_iterator(),
                  validation_data = validation_dataset.make_one_shot_iterator(),
                  callbacks=callbacks,
                  steps_per_epoch=num_train_batches,
                  validation_steps=num_validation_batches,
                  epochs=2000)

		model.save(os.path.join(self.output_path, "model.h5"))

		# os.system('gsutil cp model_weights.h5 ' + self.output_path)
		# os.system('gsutil cp model.h5 ' + self.output_path)
		# os.system('gsutil cp best.hdf5 ' + self.output_path)
		# os.system('gsutil cp latest.h5 ' + self.output_path)

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
		if self.model_path is not None:
			model = load_model(self.model_path)
		else:
			model = self.set_up_model()
		self.train(model)
