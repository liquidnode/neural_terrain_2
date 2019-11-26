import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import numpy as np

class DestroyLayer(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 1:
       raise Exception('must have exactly one inputs')
    if len(top) != 1:
       raise Exception('must have exactly one output')
  def reshape(self,bottom,top):
      self.labels = np.ones((bottom[0].shape[0], 1), dtype=np.int)
      top[0].reshape(bottom[0].shape[0], 1)
  def forward(self,bottom,top):
      top[0].data[...] = self.labels
  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass