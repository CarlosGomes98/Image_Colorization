from keras.utils import Sequence
import numpy as np
from skimage import color
from model.utilities import read_image, bucketize_gaussian
class DataGenerator(Sequence):

    def __init__(self, list_IDs, labels, path, buckets, batch_size=32, dim=(32,32,32), n_channels=313, shuffle=True):
        #Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.path = path
        self.buckets = buckets

    def on_epoch_end(self):
        #updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim))
        Y = np.empty((self.batch_size, *self.dim, self.n_channels))

        for i, ID in enumerate(list_IDs_temp):
            image = read_image(ID, self.path)
            image = image * (1./255)
            image = color.rgb2lab(image)
            X[i,] = image[:, :, 0]

            Y[i,] = bucketize_gaussian(image[:, :, 1], self.buckets)
            
        return X, Y.reshape(self.dim[0]*self.dim[1], self.n_channels)

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y