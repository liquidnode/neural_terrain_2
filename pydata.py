import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import numpy as np

class PyData(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 0:
       raise Exception('must have exactly zero inputs')
    if len(top) != 1:
       raise Exception('must have exactly one output')
    self.mode = self.param_str
  def reshape(self,bottom,top):
      if self.mode == 'a':
          top[0].reshape(1,160,10,10)
      if self.mode == 'b1':
          top[0].reshape(1,1,40,40)
      if self.mode == 'b2':
          top[0].reshape(1,1,80,80)
      if self.mode == 'b3':
          top[0].reshape(1,1,160,160)
      if self.mode == 'b4':
          top[0].reshape(1,1,200,200)
      if self.mode == 'c':
          top[0].reshape(1,80,12,12)
      if self.mode == 'd':
          top[0].reshape(1,40,40,40)
      if self.mode == 'e':
          top[0].reshape(1,200,4,4)
  def forward(self,bottom,top):
      pass
  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass