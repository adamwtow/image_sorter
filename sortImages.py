#!/usr/bin/env python

import caffeInterface

import cv2
import numpy as np
import time
import os
import random
import sys

from subprocess import call
from sklearn.cluster import KMeans

caffe_root = '/home/apc/co/caffe/'
data_dir = '/home/apc/co/image_sorter/data/UnlabelledImageProposals/'
output_dir = '/home/apc/co/image_sorter/data/output/'
model_root = '/home/apc/co/image_sorter/models/bvlc_alexnet/'
model_filename = 'bvlc_alexnet.caffemodel'
deploy_prototxt = 'deploy.prototxt'
sort_layer = 'fc7'

number_of_classes = 42

# TODO

# Delete images with (copy) in name

# Load model
caffe_model_file = model_root + deploy_prototxt
caffe_pretrained_file = model_root + model_filename
mean_vec = np.array([112.817125, 110.58435, 110.7852])

# if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
#     print 'CaffeNet found.'
# else:
#     print 'Downloading pre-trained CaffeNet model...'
#     call(['./download_model_binary.py', '/home/apc/co/image_sorter/models/bvlc_reference_caffenet'])

mynet = caffeInterface.CaffeNet(caffe_root, caffe_model_file, caffe_pretrained_file, caffe_mean_vec = mean_vec)

# I think this assumes batch size of 1
# all_features = np.zeros([1, mynet.net.blobs[sort_layer].data[0].shape[0]])
all_features = []
#print all_features


def doClassify(img):
    img_trans = mynet.transformer.preprocess(mynet.input_name, img)
    mynet.net.blobs[mynet.input_name].data[...] = img_trans
    # return mynet.net.forward()['prob'][0,]
    return mynet.net.forward()

i = 0

all_filenames = []

# Loop over all proposals
for filename in os.listdir(data_dir):
    if filename.endswith(".png"):
        imagePath = os.path.join(data_dir, filename)
        all_filenames.append(filename)

        proposal = cv2.imread(imagePath)
        doClassify(proposal)

        #print dir(mynet)
        # for each layer, show the output shape
        #for layer_name, blob in mynet.net.blobs.iteritems():
            #    print layer_name + '\t' + str(blob.data.shape)

        # Layer activations indexed by layer name - index 0 for weights/activations and 1 for biases
        features = mynet.net.blobs[sort_layer].data[0]
        # print features.shape
        # features = features[:,0,0]
        #print features.shape
        #print features
        # Fill all_features from sort_layer
        #all_features = np.concatenate((all_features, [features]), axis=0)
        #all_features = np.append(all_features, [features], axis=0)
        all_features.append(features.copy())

        #i = i + 1
        #if i == 3:
        #    break

#print all_features

# TODO: modify Lloyd's k-means algorithm to allow for keeping track of the data that I am sorting

# Note: temporary solution is to use sklearn implementation and then predict the label of each data point after clustering. Slower but works.

#print np.array(all_features)

kmeans = KMeans(n_clusters=number_of_classes, random_state=0).fit(np.array(all_features))
#print kmeans.labels_

#print 'yooooooooooooooooo'

#print kmeans.predict(all_features[0])
#print kmeans.predict(all_features[1])
#print kmeans.predict(all_features[2])

#mu,clusters = find_centers(all_features, number_of_classes)
#print clusters

for filename,feature in zip(all_filenames,all_features):
    imagePath = os.path.join(data_dir, filename)
    proposal = cv2.imread(imagePath)
    label = kmeans.predict(feature)
    output_filename = 'label_' + str(label[0]) + '_' + filename
    imagePath = os.path.join(output_dir, output_filename)
    cv2.imwrite(imagePath, proposal)
    #print imagePath
