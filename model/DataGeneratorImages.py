from keras.utils import Sequence
import numpy as np
import sklearn.neighbors as nn
from skimage.io import imread
from skimage.color import rgb2lab
from skimage.transform import resize

def soft_encode_bucketize(image_ab, nearest_neighbors, rebalance):
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
    if not rebalance is None:
        y = y * rebalance[np.argmax(y, axis=1)][:, np.newaxis]
    return y

def one_hot_bucketize(image_ab, buckets):
    #calculate the distances from each pixel to each bucket
    distances = np.zeros((image_ab.shape[0]*image_ab.shape[0], buckets.shape[0]))
    distances = cdist(image_ab.reshape(image_ab.shape[0]*image_ab.shape[0], 2), buckets)
    closest_buckets = np.argmin(distances, axis=1).reshape(image_ab.shape[0], image_ab.shape[0]).astype(int)
    identity = np.identity(buckets.shape[0])
    bucketized = np.zeros(image_ab.shape[0], image_ab.shape[1], buckets.shape[0]))
    bucketized = identity[closest_buckets]
    return bucketized.reshape(bucketized.shape + (1,))

class DataGenerator(Sequence):

    def __init__(self, images, buckets, rebalance, batch_size=64, dim=(64,64), shuffle=True):
        #Initialization
        self.dim = dim
        self.images = images
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buckets = buckets
        self.rebalance = rebalance
        self.nn = nn.NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(buckets)
        self.on_epoch_end()

    def on_epoch_end(self):
        #updates indexes after each epoch
        self.indexes = np.arange(len(self.images))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, list_IDs_temp):
        # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        X = np.empty((self.batch_size, *self.dim, 1))
        Y = np.empty((self.batch_size, self.dim[0]*self.dim[1], 313))

        for i, ID in enumerate(list_IDs_temp):
            # encoded_image = np.load(self.path + "/" + ID)
            # print(ID)
            image = resize(imread(ID), (*self.dim, 3), mode="reflect", anti_aliasing=True)
            image = rgb2lab(image)            
            L = image[:, :, 0]
            L = L - 50
            L = L / 50
            L = L.reshape(L.shape+(1,))
            X[i,] = L

            # if rebalance is None:
            #     Y[i,] = one_hot_bucketize(image[:, :, 1:], self.buckets)
            # else:
            #     Y[i,] = soft_encode_bucketize(image[:, :, 1:], self.nn, self.rebalance)
            Y[i,] = soft_encode_bucketize(image[:, :, 1:], self.nn, self.rebalance)
        
        return X, Y

    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))

    def __getitem__(self, index):
        # Generate one batch of data

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.images[k] for k in indexes]

        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

if __name__ == '__main__':
    pass