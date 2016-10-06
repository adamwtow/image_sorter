# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 11:17:34 2014

@author: niko
"""

import numpy as np
#import caffe
import imp
import copy
import time

class CaffeNet:
  """This support the newer Caffe Python interface."""

  def __init__(self, caffe_root=None, caffe_model_file=None, caffe_pretrained_file=None, caffe_mean_file=None, caffe_mean_vec=None):
    if caffe_root == None:
      caffe_root = '/home/niko/src/caffe/'

    if caffe_model_file == None:
      caffe_model_file = caffe_root + 'examples/imagenet/imagenet_deploy.prototxt'

    if caffe_pretrained_file == None:
      caffe_pretrained_file = caffe_root + 'examples/imagenet/caffe_reference_imagenet_model'

    if caffe_mean_file == None:
      caffe_mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'


    try:
      print 'Loading Caffe module from', caffe_root,'...'
      filename, path, desc =  imp.find_module('caffe', [caffe_root+'/python/'])
      self.caffe = imp.load_module('caffe', filename, path, desc)

      self.caffe.set_mode_gpu()

      print 'Loading network from definition', caffe_model_file,'and network parameters', caffe_pretrained_file
      self.net = self.caffe.Net(caffe_model_file, caffe_pretrained_file, self.caffe.TEST)

      self.blob_names = self.net.blobs.keys()
      self.input_name = self.blob_names[0]
      self.output_name = self.blob_names[-1]

      # set up transformer object
      self.init_transformer(caffe_mean_vec, caffe_mean_file)

    except ImportError:
      print "\nError: Module caffe not found at path " + caffe_root + ". I try to continue, but you cannot use Caffe's functionality."
      self.net=None
    except Exception, e:
      print "Error:", e
      self.net=None

  # ========================================================================
  def init_transformer(self, caffe_mean_vec=None, caffe_mean_file=None, blob_name=None):
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]

    if blob_name==None:
      blob_name = self.input_name

    self.transformer = self.caffe.io.Transformer({blob_name: self.net.blobs[blob_name].data.shape})
    self.transformer.set_transpose(blob_name, (2,0,1))
    if caffe_mean_vec==None:
      self.transformer.set_mean(blob_name, np.load(caffe_mean_file).mean(1).mean(1)) # mean pixel
    else:
      self.transformer.set_mean(blob_name, np.array(caffe_mean_vec)) # mean pixel
    # self.transformer.set_raw_scale(blob_name, 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    self.transformer.set_channel_swap(blob_name, (2,1,0))  # the reference model has channels in BGR order instead of RGB
