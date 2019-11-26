import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import cv2
import math
import numpy as np

class RandomBlurData(caffe.Layer):
  def setup(self, bottom, top):
    self.iter = 0
    if len(bottom) != 1:
       raise Exception('must have exactly one inputs')
    if len(top) != 1:
       raise Exception('must have exactly two output')
    try:
        self.outw = int(self.param_str)
    except ValueError:
        raise ValueError("Parameter string must be a legible int")
  def reshape(self,bottom,top):
    self.labels = np.zeros((bottom[0].shape[0], 1), dtype=np.float)
    top[0].reshape(*bottom[0].data.shape)
    #top[1].reshape(bottom[0].data.shape[0], 1)
  def forward(self,bottom,top):
    self.iter += 5
    if self.iter % 100 == 0:
        print "iter " + str(self.iter)
    start = min((self.outw + 1.) * (self.iter / 10000.), (self.outw + 1.) * (3650. / 10000.))
    end = (self.outw + 1.)
    self.labels = np.random.random_sample((bottom[0].shape[0], 1)) * (end - start) + start#(self.outw + 1)
    for i in range(top[0].shape[0]):
        wid = int(math.floor(self.labels[i, 0]))
        if wid > self.outw:
            wid = self.outw
        if wid != 0:
            top[0].data[i,0,:,:] = cv2.blur(bottom[0].data[i,0,:,:],(wid, wid))# * (1.0 + (self.labels[i, 0] / 13.0) * 0.13) # davor 0.08
        else:
            top[0].data[i,0,:,:] = bottom[0].data[i,0,:,:]
    #top[1].data[...] = self.labels
  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass


