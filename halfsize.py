import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import cv2
import math
import numpy as np
from cv2 import INTER_NEAREST

class HalfData(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 1:
       raise Exception('must have exactly one inputs')
    if len(top) != 1:
       raise Exception('must have exactly one output')
  def reshape(self,bottom,top):
    top[0].reshape(bottom[0].data.shape[0], bottom[0].data.shape[1], bottom[0].data.shape[2] / 2, bottom[0].data.shape[3] / 2)
  def forward(self,bottom,top):
    for i in range(top[0].shape[0]):
        for j in range(top[0].shape[1]):
            top[0].data[i,j,:,:] = cv2.resize(bottom[0].data[i,j,:,:],(bottom[0].data.shape[2] / 2, bottom[0].data.shape[3] / 2), interpolation=INTER_NEAREST)
  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass