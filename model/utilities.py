import numpy as np
# from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
# from keras.layers import Conv2D
from skimage import io, color, transform
import sklearn.neighbors as nn
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import tensorflow as tf
# from google.cloud import storage
import os
image_size = 128
buckets = np.load("model/pts_in_hull.npy")
rebalance = np.load("model/rebalance.npy")
nearest_neighbors = nn.NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(buckets)
# def read_image(img_id, dir):
#     try:
#         img = load_img(dir + "/" + img_id, target_size=(image_size, image_size))
#         img = img_to_array(img)
#         return img
#     except:
#         return None

def show_image(image):
    plt.imshow(image/255.)
    plt.show()

def rgb2lab_32(image):
    return color.rgb2lab(image).astype(np.float32)

def parse_function(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, ratio=2, channels=3)
    image_lab = tf.py_func(rgb2lab_32, [image], tf.float32)
    image_L = tf.py_func(preprocess_and_return_X, [image_lab], tf.float32)
    image_L = tf.reshape(image_L, [128, 128, 1])
    image_bucketized = tf.py_func(soft_encode_bucketize, [image_lab], tf.float32)
    image_bucketized = tf.reshape(image_bucketized, [128*128, 313])
    return image_L, image_bucketized

def mse_parse_function(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image_lab = tf.py_func(rgb2lab_32, [image], tf.float32)
    image_lab = tf.reshape(image_lab, [256, 256, 3])
    image_L = tf.py_func(preprocess_and_return_X, [image_lab], tf.float32)
    image_L = tf.reshape(image_L, [256, 256, 1])
    image_ab = image_lab[:, :, 1:]/128
    return image_L, image_ab

def mse_parse_function_gan(filename):
    image_string = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_string, ratio=2, channels=3)
    image_lab = tf.py_func(rgb2lab_32, [image], tf.float32)
    image_lab = tf.reshape(image_lab, [128, 128, 3])
    image_L = tf.py_func(preprocess_and_return_X, [image_lab], tf.float32)
    image_L = tf.reshape(image_L, [128, 128, 1])
    image_ab = image_lab[:, :, 1:]/128
    return image_L, image_ab

def printOutput(file, output):
    for y in range(image_size):
        row = ""
        for x in range(image_size):
            row += str(output[y, x]) + " "
        file.write(row + "\n")
    file.close()

def preprocess_and_return_X_batch(images):
    X = images[:, :, :, 0]
    X = X - 50
    X = X/50
    X = X.reshape(X.shape+(1,)).astype(np.float32)
    return X

def preprocess_and_return_X(image):
    X = image[:, :, 0]
    X = X - 50
    X = X/50
    X = X.reshape(X.shape+(1,))
    return X

# def convLayer(input, filters, kernel_size, dilation=1, stride=1):
#     return Conv2D(filters, kernel_size, padding="same", activation="relu", dilation_rate=dilation, strides=stride)(input)

# def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
#     """Lists all the blobs in the bucket that begin with the prefix.
#     This can be used to list all blobs in a "folder", e.g. "public/".
#     The delimiter argument can be used to restrict the results to only the
#     "files" in the given "folder". Without the delimiter, the entire tree under
#     the prefix is returned. For example, given these blobs:
#         /a/1.txt
#         /a/b/2.txt
#     If you just specify prefix = '/a', you'll get back:
#         /a/1.txt
#         /a/b/2.txt
#     However, if you specify prefix='/a' and delimiter='/', you'll get back:
#         /a/1.txt
#     """
#     storage_client = storage.Client()
#     bucket = storage_client.get_bucket(bucket_name)

#     blobs = bucket.list_blobs(prefix=prefix, delimiter=delimiter)

#     result = [blob.name for blob in blobs]
#     return result

def soft_encode_bucketize(image):
    image_ab = image[:, :, 1:]
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
    return y.astype(np.float32)

def decode_bucketize_images(images_to_bucketize, rebalance, batch_size):
    closest_buckets = images_to_bucketize[:, :, :, 1].astype(int)
    identity = np.identity(313)
    bucketized = np.zeros((batch_size, closest_buckets.shape[1], closest_buckets.shape[2], 313))
    bucketized = identity[closest_buckets]
    bucketized = bucketized * np.expand_dims(rebalance[np.argmax(bucketized, axis=3)], axis=3)
    return bucketized

def bucketize_gaussian_batch(imagesAB, buckets, batch_size):
    #calculate the distances from each pixel to each bucket
    distances = np.zeros((batch_size*image_size*image_size, buckets.shape[0]))
    distances = cdist(imagesAB.reshape(batch_size*image_size*image_size, 2), buckets)
    #find five shortest ones
    shortest_distances_indices = np.argpartition(distances, 5)
    five_shortest_distances_indices = shortest_distances_indices[:, :5]
    not_five_shortest_distances_indices = np.argpartition(distances, 5)[:, 5:]
    #zero the others
    vertical_indices = np.arange(batch_size*image_size*image_size)[:, np.newaxis]
    distances[vertical_indices, not_five_shortest_distances_indices] = 0
    #pass gaussian kernel and normalize 5 shortest distances
    weights = np.exp(-distances[vertical_indices, five_shortest_distances_indices]**2/(2*5**2))
    weights_norm = weights/np.sum(weights, axis=1, keepdims=True)
    distances[vertical_indices, five_shortest_distances_indices] = weights_norm
    return distances.reshape(batch_size, image_size, image_size, 313)

def decode_bucketize(image, rebalance):
    identity = np.identity(313).astype(float)
    bucketized = np.zeros((image_size, image_size, 313))
    bucketized = identity[image[:, :, 1]]
    bucketized = bucketized * np.expand_dims(rebalance[np.argmax(bucketized, axis=2)], axis=2)
    return bucketized.reshape(image_size*image_size, 313)

def decode_bucketize_no_rebalance(image):
    identity = np.identity(313).astype(float)
    bucketized = np.zeros((image_size, image_size, 313))
    bucketized = identity[image[:, :, 1]]
    return bucketized.reshape(image_size*image_size, 313)

def decode_bucketize_batch(images, rebalance):
    identity = np.identity(313).astype(float)
    bucketized = np.zeros((images.shape[0], image_size, image_size, 313))
    bucketized = identity[images[... , 1]]
    bucketized = bucketized * np.expand_dims(rebalance[np.argmax(bucketized, axis=3)], axis=3)
    return bucketized.reshape(images.shape[0], image_size*image_size, 313)

def bucketize_gaussian(imageAB, buckets, rebalance):
    #calculate the distances from each pixel to each bucket
    distances = np.zeros((image_size*image_size, buckets.shape[0]))
    distances = cdist(imageAB.reshape(image_size*image_size, 2), buckets)
    #find five shortest ones
    shortest_distances_indices = np.argpartition(distances, 5)
    five_shortest_distances_indices = shortest_distances_indices[:, :5]
    not_five_shortest_distances_indices = np.argpartition(distances, 5)[:, 5:]
    #zero the others
    vertical_indices = np.arange(image_size*image_size)[:, np.newaxis]
    distances[vertical_indices, not_five_shortest_distances_indices] = 0
    #pass gaussian kernel and normalize 5 shortest distances
    weights = np.exp(-distances[vertical_indices, five_shortest_distances_indices]**2/(2*5**2))
    weights_norm = weights/np.sum(weights, axis=1, keepdims=True)
    distances[vertical_indices, five_shortest_distances_indices] = weights_norm
    distances = distances * np.expand_dims(rebalance[np.argmax(distances, axis=1)], axis=1)
    return distances