import keras
import numpy as np
from skimage import color
from model.utilities import preprocess_and_return_X_batch, decode_bucketize_batch
import tensorflow as tf
class DataGenerator(keras.utils.Sequence):

    def __init__(self, path, num_batches, buckets, rebalance, batch_size=32, dim=(128,128), n_channels=313, shuffle=True):
        #Initialization
        self.dim = dim
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path = path
        self.buckets = buckets
        self.rebalance = rebalance
        self.on_epoch_end()

    def on_epoch_end(self):
        #updates indexes after each epoch
        self.indexes = np.arange(self.num_batches)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        # Generate one batch of data
        print("Generated batch" + str(index))
        # f = BytesIO(file_io.read_file_to_string(self.path + "/batch_" + str(index+1) + ".npy", binary_mode=True))
        # encoded_images = np.load(f)
        encoded_images = np.load(self.path + "/batch_" + str(index+1) + ".npy")
        X = preprocess_and_return_X_batch(encoded_images)

        Y = decode_bucketize_batch(encoded_images, self.rebalance)

        return X, Y
    
    # Unused
    # def __data_generation(self, list_IDs_temp):
    #     # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    #     X = np.empty((self.batch_size, *self.dim, 1))
    #     Y = np.empty((self.batch_size, self.dim[0]*self.dim[1], self.n_channels))

    #     for i, ID in enumerate(list_IDs_temp):
    #         # encoded_image = np.load(self.path + "/" + ID)
    #         f = BytesIO(file_io.read_file_to_string(self.path + "/" + ID, binary_mode=True))
    #         encoded_image = np.load(f)
    #         X[i,] = preprocess_and_return_X(encoded_image)

    #         Y[i,] = decode_bucketize(encoded_image, self.rebalance)
    #     return X, Y

    def __len__(self):
        return self.num_batches

    # def __getitem__(self, index):
    #     # Generate one batch of data

    #     indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    #     list_IDs_temp = [self.list_IDs[k] for k in indexes]

    #     X, Y = self.__data_generation(list_IDs_temp)

    #     return X, Y
