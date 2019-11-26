import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import numpy as np

class GramLayer(caffe.Layer):
  def setup(self, bottom, top):
    if len(bottom) != 1:
       raise Exception('must have exactly one input')
    if len(top) != 1:
       raise Exception('must have exactly one output')
  def reshape(self,bottom,top):
    top[0].reshape(bottom[0].shape[0], 1, bottom[0].shape[1], bottom[0].shape[1])
  def forward(self,bottom,top):
    N = bottom[0].shape[1]
    for m in range(0, bottom[0].shape[0]):
        F = bottom[0].data[m,:,:,:].reshape(N,-1)
        M = F.shape[1]
        G = np.dot(F,F.T) / (2.0 * M**2 * N**2)
        top[0].data[m,0,:,:] = G
  def backward(self,top,propagate_down,bottom):
    N = bottom[0].shape[1]
    for m in range(0, bottom[0].shape[0]):
        F = bottom[0].data[m,:,:,:].reshape(N,-1)
        M = F.shape[1]
        G = top[0].diff[m,:,:,:].reshape(N,-1)
        bottom[0].diff[m,:,:,:] = (np.dot(G,F) * (F>0) / (M**2 * N**2)).reshape(bottom[0].shape[1],bottom[0].shape[2],bottom[0].shape[3])