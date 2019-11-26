import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import cv2
import numpy as np

class ODummyData(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 0:
       raise Exception('must have exactly zero inputs')
    if len(top) != 2:
       raise Exception('must have exactly two output')
    try:
        self.batchsize = int(self.param_str)
    except ValueError:
        raise ValueError("Parameter string must be a legible int")
  def reshape(self,bottom,top):
    top[0].reshape(self.batchsize, self.batchsize)
    self.labels = np.ones((self.batchsize, 1), dtype=np.int)
    top[1].reshape(self.batchsize, 1)
    self.output = np.zeros((self.batchsize, self.batchsize), dtype=np.float32)
    for i in range(self.batchsize):
        self.output[i,i] = 1.0
  def forward(self,bottom,top):
    top[0].data[...] = self.output
    top[1].data[...] = self.labels
  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass