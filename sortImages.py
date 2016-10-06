#!/usr/bin/env python

import caffeInterface

import cv2
import numpy as np
import time
import os
import random

from sklearn.cluster import KMeans

data_dir = '/home/apc/co/image_sorter/data/UnlabelledImageProposals/'
output_dir = '/home/apc/co/image_sorter/data/output/'
model_root = '/home/apc/co/image_sorter/models/'
caffe_root = '/home/apc/co/caffe/'
snapshot_filename = 'chris_iter_2500.caffemodel'
sort_layer = 'pool5/7x7_s1'

# Load model
caffe_model_file = model_root + 'deploy_picking.prototxt'
caffe_pretrained_file = model_root + snapshot_filename
mean_vec = np.array([112.817125, 110.58435, 110.7852])
GoogLeNet = caffeInterface.CaffeNet(caffe_root, caffe_model_file, caffe_pretrained_file, caffe_mean_vec = mean_vec)

# I think this assumes batch size of 1
# all_features = np.zeros([1, GoogLeNet.net.blobs[sort_layer].data[0].shape[0]])
all_features = []
#print all_features


def doClassify(img):
    img_trans = GoogLeNet.transformer.preprocess(GoogLeNet.input_name, img)
    GoogLeNet.net.blobs[GoogLeNet.input_name].data[...] = img_trans
    # return GoogLeNet.net.forward()['prob'][0,]
    return GoogLeNet.net.forward()

def cluster_points(X, mu):
    clusters  = {}
    for x in X:
        bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)


i = 0

all_filenames = []

# Loop over all proposals
for filename in os.listdir(data_dir):
    if filename.endswith(".png"):
        imagePath = os.path.join(data_dir, filename)
	all_filenames.append(filename)

        proposal = cv2.imread(imagePath)
        doClassify(proposal)

	#print dir(GoogLeNet)
	# for each layer, show the output shape
	#for layer_name, blob in GoogLeNet.net.blobs.iteritems():
    	#    print layer_name + '\t' + str(blob.data.shape)

	# Layer activations indexed by layer name - index 0 for weights/activations and 1 for biases
	features = GoogLeNet.net.blobs[sort_layer].data[0]
	#print features.shape
	features = features[:,0,0]
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

number_of_classes = 42

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
