import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import cv2
import numpy as np

class ChooseData(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 1:
       raise Exception('must have exactly one inputs')
    if len(top) != 1:
       raise Exception('must have exactly one output')
    try:
        self.batchsize = int(self.param_str)
    except ValueError:
        raise ValueError("Parameter string must be a legible int")
  def reshape(self,bottom,top):
    top[0].reshape(self.batchsize, bottom[0].data.shape[1], bottom[0].data.shape[2], bottom[0].data.shape[3])
  def forward(self,bottom,top):
    for i in range(self.batchsize):
        rand = np.random.random_integers(0, bottom[0].data.shape[0]-1)
        top[0].data[i,:,:,:] = bottom[0].data[rand,:,:,:]
  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass