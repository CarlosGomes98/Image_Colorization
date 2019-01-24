import numpy as np
import os
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
    bucketized = np.zeros((image_ab.shape[0], image_ab.shape[1], buckets.shape[0]))
    bucketized = identity[closest_buckets]
    return bucketized.reshape(bucketized.shape + (1,))

rebalance = np.load("model/rebalance.npy")
nn = nn.NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(np.load("model/pts_in_hull.npy"))
X = np.empty((16, 128, 128, 1))
Y = np.empty((16, 128*128, 313))
counter = 0
batch = 0
for index, image_name in enumerate(os.listdir("data/Validation")):
    counter = counter + 1
    image = resize(imread("data/Validation/"+image_name), (128, 128, 3), mode="reflect", anti_aliasing=True)
    image = rgb2lab(image)            
    L = image[:, :, 0]
    L = L - 50
    L = L / 50
    L = L.reshape(L.shape+(1,))
    X[index%16,] = L

    Y[index%16,] = soft_encode_bucketize(image[:, :, 1:], nn, None) #rebalance goes here

    if counter%16 == 0:
        np.save("data/Validation_np/"+str(batch)+"_X.npy", X.astype(np.float16))
        np.save("data/Validation_np/"+str(batch)+"_Y.npy", Y.astype(np.float16))
        batch = batch + 1
        counter = 0

