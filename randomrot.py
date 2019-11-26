import sys, os
#sys.path.insert(0,os.environ['CAFFE_ROOT'] + '/python')
import caffe
import cv2
import math
import numpy as np
import random
from math import sin
from math import cos

class RandomRotLayer(caffe.Layer):
  def subimage(self,image, center, theta, width, height):
    #theta *= 3.14159 / 180 # convert to rad

    v_x = (cos(theta), sin(theta))
    v_y = (-sin(theta), cos(theta))
    s_x = center[0] - v_x[0] * (width / 2) - v_y[0] * (height / 2)
    s_y = center[1] - v_x[1] * (width / 2) - v_y[1] * (height / 2)

    mapping = np.array([[v_x[0],v_y[0], s_x],
                        [v_x[1],v_y[1], s_y]])

    return cv2.warpAffine(image,mapping,(width, height),flags=cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
  def setup(self, bottom, top):
    if len(bottom) != 1:
       raise Exception('must have exactly one inputs')
    if len(top) != 1:
       raise Exception('must have exactly two output')
    try:
        self.outw = int(self.param_str)
    except ValueError:
        raise ValueError("Parameter string must be a legible int")
  def reshape(self,bottom,top):
    top[0].reshape(bottom[0].data.shape[0], 1, self.outw, self.outw)#*bottom[0].data.shape)
  def forward(self,bottom,top):
    for i in range(top[0].data.shape[0]):
        angle = random.uniform(0, 3.14159 * 2.)
        boarder = int(((abs(sin(angle)*self.outw) + abs(cos(angle)*self.outw))/2.) + 0.5)
        ox = random.randint(boarder, bottom[0].data.shape[2]-boarder)
        oy = random.randint(boarder, bottom[0].data.shape[3]-boarder)
        top[0].data[i, 0, :, :] = self.subimage(bottom[0].data[i,0,:,:], [ox, oy], angle, self.outw, self.outw)
  def backward(self,top,propagate_down,bottom):
    # no back prop
    pass


