import keras
import numpy as np
from io import BytesIO
from tensorflow.python.lib.io import file_io

def soft_encode_bucketize(image_ab, nearest_neighbors):
    h, w = image_ab.shape[:2]
    a = np.ravel(image_ab[:, :, 0])
    b = np.ravel(image_ab[:, :, 1])
    #h*w, 2 matrix where each row is an ab pair corresponding to a pixel
    ab = np.vstack((a, b)).T
    # Get the distance to and the idx of the nearest neighbors
    dist_neighbors, idx_neighbors = nearest_neighbors.kneighbors(ab)
    # Smooth the weights with a gaussian kernel
    sigma = 5
    weights = np.exp(-dist_neighbors ** 2 / (2 * sigma ** 2))
    weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
    # format the tar get
    y = np.zeros((ab.shape[0], 313))
    indeces = np.arange(ab.shape[0])[:, np.newaxis]
    y[indeces, idx_neighbors] = weights
    y = y.reshape(h, w, 313)
    return y

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
        self.on_epoch_end()
        if rebalance is None:
            self.identity = np.identity(313).astype(float)
        else:
            self.identity =  np.identity(313).astype(float) * self.rebalance

    def on_epoch_end(self):
        #updates indexes after each epoch
        self.indexes = np.arange(self.num_batches)
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        # Generate one batch of data
        print("Generated batch" + str(index))
        f = BytesIO(file_io.read_file_to_string(self.path + "/batch_" + str(index+1) + ".npy", binary_mode=True))
        encoded_images = np.load(f)
        # encoded_images = np.load(self.path + "/batch_" + str(index+1) + ".npy")
        X = encoded_images[:, :, :, 0]
        X = X - 50
        X = X/50
        X = X.reshape(X.shape+(1,))

        bucketized = np.zeros((encoded_images.shape[0], *self.dim, 313))
        bucketized = self.identity[encoded_images[... , 1]]
        Y = bucketized.reshape(encoded_images.shape[0], self.dim[0]*self.dim[1], 313)

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
if __name__ == '__main__':
    split_data()