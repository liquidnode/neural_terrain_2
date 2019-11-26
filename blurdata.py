import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import cv2
import numpy as np

class BlurData(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 1:
       raise Exception('must have exactly one inputs')
    if len(top) != 1:
       raise Exception('must have exactly one output')
    try:
        self.outw = int(self.param_str)
    except ValueError:
        raise ValueError("Parameter string must be a legible int")
  def reshape(self,bottom,top):
    top[0].reshape(*bottom[0].data.shape)
  def forward(self,bottom,top):
    for i in range(top[0].shape[0]):
        top[0].data[i,0,:,:] = cv2.blur(bottom[0].data[i,0,:,:],(self.outw, self.outw))# * (1.0 + (self.outw / 13.0) * 0.13)
        #top[0].data[i,0,:,:] = cv2.resize(bottom[0].data[i,0,:,:],(self.outw, self.outw), interpolation = cv2.INTER_LINEAR)
  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass