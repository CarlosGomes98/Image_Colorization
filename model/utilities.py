import numpy as np
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.layers import Conv2D
from skimage import io, color, transform
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
import os
image_size = 128

def read_image(img_id, dir):
    try:
        img = load_img(dir + "/" + img_id, target_size=(image_size, image_size))
        img = img_to_array(img)
        return img
    except:
        return None

def show_image(image):
    plt.imshow(image/255.)
    plt.show()

def printOutput(file, output):
    for y in range(image_size):
        row = ""
        for x in range(image_size):
            row += str(output[y, x]) + " "
        file.write(row + "\n")
    file.close()

def preprocess_and_return_X(examples):
    examples = examples[:, :, :, 0]
    examples = examples - 50
    examples = examples/50
    examples = examples.reshape(examples.shape+(1,))
    return examples

def convLayer(input, filters, kernel_size, dilation=1, stride=1):
    return Conv2D(filters, kernel_size, padding="same", activation="relu", dilation_rate=dilation, strides=stride)(input)

def bucketize_images(images_to_bucketize, batch_size):
    closest_buckets = images_to_bucketize[:, :, :, 1].astype(int)
    identity = np.identity(313)
    bucketized = np.zeros((batch_size, closest_buckets.shape[1], closest_buckets.shape[2], 313))
    bucketized = identity[closest_buckets]
    return bucketized

def bucketize_gaussian(imagesAB, buckets, batch_size):
    #calculate the distances from each pixel to each bucket
    distances = np.zeros((batch_size*image_size*image_size, buckets.shape[0]))
    distances = cdist(imagesAB.reshape(batch_size*image_size*image_size, 2), buckets)
    #find five shortest ones
    shortest_distances_indices = np.argpartition(distances, 5)
    five_shortest_distances_indices = shortest_distances_indices[:, :4]
    not_five_shortest_distances_indices = np.argpartition(distances, 5)[:, 5:]
    #zero the others
    vertical_indices = np.arange(batch_size*image_size*image_size)[:, np.newaxis]
    distances[vertical_indices, not_five_shortest_distances_indices] = 0
    #pass gaussian kernel and normalize 5 shortest distances
    weights = np.exp(-distances[vertical_indices, five_shortest_distances_indices]**2/(2*5**2))
    weights_norm = weights/np.sum(weights, axis=1, keepdims=True)
    distances[vertical_indices, five_shortest_distances_indices] = weights_norm
    return distances.reshape(batch_size, image_size, image_size, 313)
