import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import cv2
import numpy as np

class SwapLayer(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 2:
       raise Exception('must have exactly two inputs')
    if len(top) != 2:
       raise Exception('must have exactly two output')
  def reshape(self,bottom,top):
    self.labels = np.zeros((bottom[0].data.shape[0], 1), dtype=np.int)
    top[0].reshape(*bottom[0].data.shape)
    top[1].reshape(bottom[0].data.shape[0], 1)
  def forward(self,bottom,top):
    self.labels = np.random.random_integers(0, 1, (bottom[0].shape[0], 1))
    for i in range(0, bottom[0].shape[0]):
        #self.labels[x][1] = 1. - self.labels[x][0]
        top[0].data[i][...] = bottom[self.labels[i][0]].data[i]
    top[1].data[...] = self.labels
  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass


