
import numpy as np
import os
import sys
import argparse
import glob
import time
from skimage import data, io, filters
from owndummy import *
from pydata import *
from gramlayer import * 
from destroy import *
from swaplayer import *
from blurdata import *
from randomblur import *
from halfsize import *
from chooselayer import *
from random import randint
import cv2
import struct

import caffe
from caffe import layers as L
from caffe import params as P
from pnoise import fractalnoise

from cv2 import INTER_CUBIC
from __builtin__ import open

inmemory = False
upsample = True
global_stat = True
allparamnames = [
        "Convolution1",
"gconvconvA0_1",
"Convolution2",
"gconvconvA0_2",
"Convolution3",
"gconvconvA0_3",


"Convolution4",
"gconvconvA1_1",
"Convolution5",
"gconvconvA1_2",
"Convolution6",
"gconvconvA1_3",


"Convolution7",
"gconvconvA2_1",
"Convolution8",
"gconvconvA2_2",
"Convolution9",
"gconvconvA2_3",


"Convolution10",
"gconvconvA3_1",
"Convolution11",
"gconvconvA3_2",
"Convolution12",
"gconvconvA3_3",


"Convolution13",
"gconvconvA4_1",
"Convolution14",
"gconvconvA4_2",
"Convolution15",
"gconvconvA4_3",


"upsamplejoinA",
"upsampleBjoinA",
"geluconvA3_3",

"Convolution16",
"gconvconvB_1",
"Convolution17",
"gconvconvB_2",
"Convolution18",
"gconvconvB_3",


"upsamplejoinB",
"upsampleBjoinB",
"geluconvA2_3",

"Convolution19",
"gconvconvC_1",
"Convolution20",
"gconvconvC_2",
"Convolution21",
"gconvconvC_3",


"upsamplejoinC",
"upsampleBjoinC",
"geluconvA1_3",

"Convolution22",
"gconvconvD_1",
"Convolution23",
"gconvconvD_2",
"Convolution24",
"gconvconvD_3",


"upsamplejoinD",
"upsampleBjoinD",
"geluconvA0_3",

"Convolution25",
"gconvconvE_1",
"Convolution26",
"gconvconvE_2",
"Convolution27",
"gconvconvE_3",

"texture",

"convl1",
"BatchNorm1",
"convl2",
"BatchNorm2",
"convl3",
"BatchNorm3",
"convl4",
"BatchNorm4",
"convl5",
"BatchNorm5",
"convl6",
"BatchNorm6",
"convl7",
"BatchNorm7",
"convl8",
"BatchNorm8",
"convl9",
"BatchNorm9",
"gangram",
"full1",
"BatchNorm10",
"full2",
"BatchNorm11",
"full3"
        ]

def convBlock(name, opn, n, input, train=True):
    trainparam = []
    trainparam2 = []
    if train:
        trainparam = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
        trainparam2 = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
    else:
        trainparam = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
        trainparam2 = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]

    convolution_param = dict(num_output=opn, kernel_size=3, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n['gconv'+name+"_1"] = L.Convolution(n[input], convolution_param=convolution_param, param=trainparam, name=allparamnames.pop(0))
    #if train:
    n['gconv'+name+"_1"] = L.BatchNorm(n['gconv'+name+"_1"], use_global_stats=global_stat, param=trainparam2, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_1"] = L.ReLU(n['gconv'+name+"_1"], negative_slope=0.1)
    n['gconv'+name+"_2"] = L.Convolution(n['gelu'+name+"_1"], convolution_param=convolution_param, param=trainparam, name=allparamnames.pop(0))
    #if train:
    n['gconv'+name+"_2"] = L.BatchNorm(n['gconv'+name+"_2"], use_global_stats=global_stat, param=trainparam2, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_2"] = L.ReLU(n['gconv'+name+"_2"], negative_slope=0.1)
    convolution_param = dict(num_output=opn, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n['gconv'+name+"_3"] = L.Convolution(n['gelu'+name+"_2"], convolution_param=convolution_param, param=trainparam, name=allparamnames.pop(0))
    #if train:
    n['gconv'+name+"_3"] = L.BatchNorm(n['gconv'+name+"_3"], use_global_stats=global_stat, param=trainparam2, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_3"] = L.ReLU(n['gconv'+name+'_3'], negative_slope=0.1)
    return n, 'gelu'+name+'_3'

def joinBlock(name, opn, n, inputA, inputB, train=True):
    trainparam = []
    trainparam2 = []
    if train:
        trainparam = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
        trainparam2 = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
    else:
        trainparam = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
        trainparam2 = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]

     #TRAINABLE???? TODO
    if upsample == False:
        convolution_param = dict(num_output=opn, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
        n["upsample"+name] = L.Deconvolution(n[inputA], convolution_param=convolution_param, param=trainparam, name=allparamnames.pop(0)) 
    else:
        convolution_param = dict(num_output=opn, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='constant', value=0.0), bias_term=False)
        n["upsample"+name] = L.Deconvolution(n[inputA], convolution_param=convolution_param
                             ,param=[dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))

    n["upsampleB"+name] = L.BatchNorm(n["upsample"+name], use_global_stats=global_stat, param=trainparam2, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n[inputB] = L.BatchNorm(n[inputB], use_global_stats=global_stat, param=trainparam2, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    #ACHTUNG BEI BATCHNORM upsampleB genau da drunter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    n["concat"+name] = L.Concat(n["upsampleB"+name], n[inputB], concat_param=dict(axis=1))
    return n, "concat"+name

def GAN(a0, batchsize, n):

    listofsizes = [a0]
    
    level = 5
    for i in range(0, level-1):
        alast = listofsizes[i]
        listofsizes.append((alast - 4)*2)
    listofsizes[0] -= 4

    n["Adata0"] = L.Input(shape=[dict(dim=[batchsize, 1, listofsizes[level-1] + 4, listofsizes[level-1] + 4])],ntop=1)

    #n["Adata0"] = L.Python(n.Adata, python_param=dict(module='blurdata', 
    #                                               layer='BlurData',
    #                                               param_str='12'))
    codings = [8, 16, 24, 32, 40]
    #codings = [8, 8, 16, 28, 40]
    
    outname = ""
    for i in range(0,level):
        n["AZrand_"+str(i)] = L.Input(shape=[dict(dim=[batchsize, 1, listofsizes[level-1 -i] + 4, listofsizes[level-1 -i] + 4])],ntop=1)
        #L.DummyData(shape=[dict(dim=[batchsize, 1, listofsizes[level-1 -i] + 4, listofsizes[level-1 -i] + 4])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)#type='gaussian', std=127.5), ntop=1)#
        n["Acdata"+str(i)] = L.Concat(n["AZrand_"+str(i)], n["Adata"+str(i)], concat_param=dict(axis=1))
        n, outname = convBlock("AconvA"+str(i), codings[0], n, "Acdata"+str(i), train=False)
        if i != level-1:
            n["Adata"+str(i+1)]= L.Input(shape=[dict(dim=[batchsize, 1, listofsizes[level-2 - i] + 4, listofsizes[level-2 - i] + 4])],ntop=1)
            #L.Python(n["Adata"+str(i)], python_param=dict(module='halfsize', 
                                 #                          layer='HalfData'))


    #start joining
    n, outname = joinBlock("AjoinA", codings[0], n, outname, 'gelu'+'AconvA3'+'_3', train=False) # oder doch 8???

    n, outname = convBlock("AconvB", codings[1], n, outname, train=False)
    n, outname = joinBlock("AjoinB", codings[1], n, outname, 'gelu'+'AconvA2'+'_3', train=False)

    n, outname = convBlock("AconvC", codings[2], n, outname, train=False)
    n, outname = joinBlock("AjoinC", codings[2], n, outname, 'gelu'+'AconvA1'+'_3', train=False)

    n, outname = convBlock("AconvD", codings[3], n, outname, train=False)
    n, outname = joinBlock("AjoinD", codings[3], n, outname, 'gelu'+'AconvA0'+'_3', train=False)

    n, outname = convBlock("AconvE", codings[4], n, outname, train=False)
    convolution_param = dict(num_output=1, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n["Atexture"] = L.Convolution(n[outname], convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))

    return n

def convBlock2(name, opn, n, input, train=True):
    trainparam = []
    trainparam2 = []
    if train:
        trainparam = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
        trainparam2 = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
    else:
        trainparam = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
        trainparam2 = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
    #if global_stat:
        #trainparam2 = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
    s = list(name)
    del s[0]
    if s[0] == 'B':
        del s[0]
    nname = "".join(s)
    convolution_param = dict(num_output=opn, kernel_size=3, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n['gconv'+name+"_1"] = L.Convolution(n[input], convolution_param=convolution_param, param=trainparam, name='gconv'+nname+"_1param")
    #if train:
    n['gconv'+name+"_1"] = L.BatchNorm(n['gconv'+name+"_1"], use_global_stats=global_stat, param=trainparam2, name='gconv'+nname+"_1param2")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_1"] = L.ReLU(n['gconv'+name+"_1"], negative_slope=0.1)

    n['gconv'+name+"_2"] = L.Convolution(n['gelu'+name+"_1"], convolution_param=convolution_param, param=trainparam, name='gconv'+nname+"_2param")
    #if train:
    n['gconv'+name+"_2"] = L.BatchNorm(n['gconv'+name+"_2"], use_global_stats=global_stat, param=trainparam2, name='gconv'+nname+"_2param2")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_2"] = L.ReLU(n['gconv'+name+"_2"], negative_slope=0.1)
    convolution_param = dict(num_output=opn, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n['gconv'+name+"_3"] = L.Convolution(n['gelu'+name+"_2"], convolution_param=convolution_param, param=trainparam, name='gconv'+nname+"_3param")
    #if train:
    n['gconv'+name+"_3"] = L.BatchNorm(n['gconv'+name+"_3"], use_global_stats=global_stat, param=trainparam2, name='gconv'+nname+"_3param2")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_3"] = L.ReLU(n['gconv'+name+'_3'], negative_slope=0.1)
    return n, 'gelu'+name+'_3'

def joinBlock2(name, opn, n, inputA, inputB, train=True):
    trainparam = []
    trainparam2 = []
    if train:
        trainparam = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
        trainparam2 = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
    else:
        trainparam = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
        trainparam2 = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
    #if global_stat:
        #trainparam2 = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
    #TRAINABLE???? TODO
    s = list(name)
    del s[0]
    if s[0] == 'B':
        del s[0]
    nname = "".join(s)
    if upsample == False:
        convolution_param = dict(num_output=opn, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
        n["upsample"+name] = L.Deconvolution(n[inputA], convolution_param=convolution_param, param=trainparam, name="upsampleparam"+nname) 
    else:
        convolution_param = dict(num_output=opn, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='constant', value=0.0), bias_term=False)
        n["upsample"+name] = L.Deconvolution(n[inputA], convolution_param=convolution_param
                             ,param=[dict(lr_mult=0, decay_mult= 0)], name="upsampleparam"+nname)

    n["upsampleB"+name] = L.BatchNorm(n["upsample"+name], use_global_stats=global_stat, param=trainparam2, name="upsampleBparam"+nname)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n[inputB] = L.BatchNorm(n[inputB], use_global_stats=global_stat, param=trainparam2, name=nname+"param")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    #ACHTUNG BEI BATCHNORM upsampleB genau da drunter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    n["concat"+name] = L.Concat(n["upsampleB"+name], n[inputB], concat_param=dict(axis=1))
    return n, "concat"+name


def LowGAN(a0, batchsize, n):
    listofsizes = [a0]
    #input w = 30 damit output 48=192/4=(12*16)/4
    level = 2
    for i in range(0, level-1):
        alast = listofsizes[i]
        listofsizes.append((alast - 4)*2)
    listofsizes[0] -= 4

    codings = [8, 16, 24, 32, 40]

    outname = ""
    for i in range(0,level):
        n["AZrand_"+str(i)] = L.Input(shape=[dict(dim=[batchsize, 1, listofsizes[level-1 -i] + 4, listofsizes[level-1 -i] + 4])],ntop=1)
        #L.DummyData(shape=[dict(dim=[batchsize, 1, listofsizes[level-1 -i] + 4, listofsizes[level-1 -i] + 4])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)
        n, outname = convBlock2("AconvA"+str(i), codings[0], n, "AZrand_"+str(i), train=False)

    n, outname = joinBlock2("AjoinA", codings[0], n, outname, 'gelu'+'AconvA0'+'_3', train=False)

    n, outname = convBlock2("AconvB", codings[1], n, outname, train=False)
    convolution_param = dict(num_output=1, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n["Atexture"] = L.Convolution(n[outname], convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="Atextureparam")
    return n



def initGENGAN(w, batchsize):
    global allparamnames
    n = caffe.NetSpec()
    n = GAN(w, batchsize, n)
    return n.to_proto()

def initLowGENGAN(w, batchsize):
    global allparamnames
    n = caffe.NetSpec()
    n = LowGAN(w, batchsize, n)
    return n.to_proto()

def copyParams(orig, gen):
    #print "start copy"
    for key in gen.params.keys():
        if key in orig.net.params:
            for i in range(0,len(gen.params[key])):
                gen.params[key][i].data[...] = orig.net.params[key][i].data
    #print "end copy"
    return gen

MAX_INT = (1<<31)-1
def IntNoise(x):
    x = int(x)
    x = ((x << 13) & MAX_INT) ^ x
    x = ( x * (x * x * 15731 + 789221) + 1376312589 ) & MAX_INT
    return 1.0 - x / 1073741824.0

def IntNoise(x, y):
    x = int(x)
    y = int(y)
    x = ((x << 13) & MAX_INT) ^ y
    y = ((y << 13) & MAX_INT) ^ x
    x = ( x * (y * x * 15731 + 789221) + 1376312589 ) & MAX_INT
    return 1.0 - x / 1073741824.0

def getNoise(strtx, strty, endx, endy, level):
    offsetsx = [28978, -189723, 500, 4090, -10029]
    offsetsy = [-3748, 281082, 73836, 0, -8192]
    outp = np.zeros((endx - strtx, endy - strty), dtype=np.float)
    for x in range(strtx, endx):
        for y in range(strty, endy):
            outp[x - strtx,y - strty] = ((IntNoise(x + offsetsx[level], y + offsetsy[level]) + 1.) / 2.) * 255.
    return outp

def getOutput(gennet, transformer, img, offsetmidx, offsetmidy):
    lstsizes = [76, 44, 28, 20, 12]
    d=1
    #last 192

    for i in range(0, 5):
        nsize = lstsizes[i] * d
        strtx = offsetmidx - (nsize / 2)
        strty = offsetmidy - (nsize / 2)
        endx = offsetmidx + (nsize / 2)
        endy = offsetmidy + (nsize / 2)
        preimg = cv2.blur(img[strtx:endx, strty:endy, 0],(12, 12))
        if d != 1:
            gennet.blobs["Adata"+str(i)].data[...] = transformer.preprocess("Adata"+str(i), cv2.resize(preimg,(lstsizes[i], lstsizes[i]), interpolation=INTER_NEAREST).reshape(lstsizes[i], lstsizes[i],1))
        else:
            gennet.blobs["Adata"+str(i)].data[...] = transformer.preprocess("Adata"+str(i), preimg.reshape(lstsizes[i], lstsizes[i],1))
        gennet.blobs["AZrand_"+str(i)].data[...] = transformer.preprocess("Adata"+str(i), getNoise(strtx / d, strty / d, endx / d, endy / d, i).reshape(lstsizes[i], lstsizes[i],1))
        d *= 2

    gennet.forward()

    return np.array(gennet.blobs["Atexture"].data[0, 0, 2:66, 2:66], copy=True)

def getFractOutput(gennet, transformer, offsetmidx, offsetmidy):
    lstsizes = [76, 44, 28, 20, 12]
    d=1
    #last 192
    img = np.zeros((lstsizes[4] * 16, lstsizes[4] * 16), dtype=np.float)

    for x in range(-lstsizes[4] * 8, lstsizes[4] * 8):
        for y in range(-lstsizes[4] * 8, lstsizes[4] * 8):
            rx = x + offsetmidx
            ry = y + offsetmidy

            img[x + lstsizes[4] * 8, y + lstsizes[4] * 8] = (fractalnoise(rx, ry, 1./(50.), 3) + 0.1) * 128.

    for i in range(0, 5):
        nsize = lstsizes[i] * d
        strtx = offsetmidx - (nsize / 2)
        strty = offsetmidy - (nsize / 2)
        endx = offsetmidx + (nsize / 2)
        endy = offsetmidy + (nsize / 2)
        boarder = (lstsizes[4] * 16 - nsize) / 2
        preimg = img[boarder:(lstsizes[4] * 16 - boarder), boarder:(lstsizes[4] * 16 - boarder)]
        #if d != 1:
        gennet.blobs["Adata"+str(i)].data[...] = transformer.preprocess("Adata"+str(i), cv2.resize(preimg,(lstsizes[i], lstsizes[i]), interpolation=INTER_NEAREST).reshape(lstsizes[i], lstsizes[i],1))
        #else:
            #gennet.blobs["Adata"+str(i)].data[...] = transformer.preprocess("Adata"+str(i), preimg.reshape(lstsizes[i], lstsizes[i],1))
        gennet.blobs["AZrand_"+str(i)].data[...] = transformer.preprocess("Adata"+str(i), getNoise(strtx / d, strty / d, endx / d, endy / d, i).reshape(lstsizes[i], lstsizes[i],1))
        d *= 2

    gennet.forward()
    boarder = (lstsizes[4] * 16 - 64) / 2
    return np.array(gennet.blobs["Atexture"].data[0, 0, 2:66, 2:66], copy=True), np.array(img[boarder:(lstsizes[4] * 16 - boarder), boarder:(lstsizes[4] * 16 - boarder)], copy=True)

def getLowOutput(gennet, transformer, lowgennet, offsetmidx, offsetmidy):
    lstsizes = [76, 44, 28, 20, 12]
    lstsizes2 = [56, 30]
    d=1
    #last 192
    for i in range(0, 2):
        nsize = lstsizes2[i] * d
        strtx = (offsetmidx/4) - (nsize / 2)
        strty = (offsetmidy/4) - (nsize / 2)
        endx = (offsetmidx/4) + (nsize / 2)
        endy = (offsetmidy/4) + (nsize / 2)
        lowgennet.blobs["AZrand_"+str(i)].data[...] = transformer.preprocess("Adata_"+str(i), getNoise(strtx / d, strty / d, endx / d, endy / d, i+2).reshape(lstsizes2[i], lstsizes2[i],1))
        d *= 2

    lowgennet.forward()
    img = cv2.resize(lowgennet.blobs["Atexture"].data[0, 0, :, :],(12*16, 12*16), interpolation=INTER_CUBIC)
    img = cv2.blur(img,(15, 15)) #15 ist groessere berge #normal 12
    d=1
    for i in range(0, 5):
        nsize = lstsizes[i] * d
        strtx = offsetmidx - (nsize / 2)
        strty = offsetmidy - (nsize / 2)
        endx = offsetmidx + (nsize / 2)
        endy = offsetmidy + (nsize / 2)
        preimg = img[(6*16-(nsize / 2)):(6*16+(nsize / 2)), (6*16-(nsize / 2)):(6*16+(nsize / 2))]#cv2.blur(img[strtx:endx, strty:endy, 0],(12, 12))
        if d != 1:
            gennet.blobs["Adata"+str(i)].data[...] = transformer.preprocess("Adata"+str(i), cv2.resize(preimg,(lstsizes[i], lstsizes[i]), interpolation=INTER_NEAREST).reshape(lstsizes[i], lstsizes[i],1))
        else:
            gennet.blobs["Adata"+str(i)].data[...] = transformer.preprocess("Adata"+str(i), preimg.reshape(lstsizes[i], lstsizes[i],1))
        gennet.blobs["AZrand_"+str(i)].data[...] = transformer.preprocess("Adata"+str(i), (getNoise(strtx / d, strty / d, endx / d, endy / d, i)*1.0).reshape(lstsizes[i], lstsizes[i],1))
        d *= 2

    gennet.forward()
    nsize=64
    return np.array(gennet.blobs["Atexture"].data[0, 0, 2:66, 2:66], copy=True), np.array(img[6*16-(nsize / 2):6*16+(nsize / 2), 6*16-(nsize / 2):6*16+(nsize / 2)], copy=True)

def main():
    pycaffe_dir = os.path.dirname(__file__)

#    caffe.set_random_seed(10923)
    caffe.set_mode_gpu();



    with open('teraGENGAN.prototxt', 'w') as f:
        n = initGENGAN(12, 1)
        f.write(str(n))

    with open('teraLowGENGAN.prototxt', 'w') as f:
        n = initLowGENGAN(30, 1)
        f.write(str(n))


    #orig = caffe.get_solver('solver_3.prototxt')
    #orig.restore('teraGenNet2_adadelta_iter_22000.solverstate')

    #start = 1000 hatte gute results
    #10000 ist auch gut
    #11500 sehr gut
    start = 2500

    solver = caffe.get_solver('solver_7.prototxt')
    resto = (start != 0)
    if resto:
        solver.restore('tera2OGAN_phase2_adadelta_iter_'+str(start)+'.solverstate')

    for i in range(0,64):
        print solver.net.params["last"][0].data[0,i]



    gennet = caffe.Net('teraGENGAN.prototxt', caffe.TEST);

    copyParams(solver, gennet)

    solver = caffe.get_solver('solver_9.prototxt')
    #ACHTUNG !!!!!!!!!!!! FALLS hier geändert in <...>.caffemodel => <...>tmp2.caffemodel
    solver.restore('teraFreeLowGAN_adadelta_iter_20000tmp2.solverstate')

    lowgennet = caffe.Net('teraLowGENGAN.prototxt', caffe.TEST);

    copyParams(solver, lowgennet)

    img = caffe.io.load_image('data/test_data_3500_1000.bmp', color=False)
    #transformer.set_mean('data',np.array([114.8],np.float32))
    width = img.shape[0]
    heigth = img.shape[1]
    #img-=114.8/255.0
    img*=255.0
    img-=103.939#/255.0
    #img-=114.8
    print width
    print heigth

    
    lstsizes = [76, 44, 28, 20, 12]

    datadict = {}
    for i in range(0, 5):
        datadict["Adata"+str(i)] = (1, 1, lstsizes[i], lstsizes[i])

    lstsizes2 = [56, 30]

    for i in range(0, 2):
        datadict["Adata_"+str(i)] = (1, 1, lstsizes2[i], lstsizes2[i])

    transformer = caffe.io.Transformer(datadict)

    for i in range(0, 5):
        transformer.set_transpose("Adata"+str(i), (2,0,1))
        transformer.set_raw_scale("Adata"+str(i), 1.0)     

    for i in range(0, 2):
        transformer.set_transpose("Adata_"+str(i), (2,0,1))
        transformer.set_raw_scale("Adata_"+str(i), 1.0) 
         
    if False:
        offsetmidx = 100
        offsetmidy = 120

        outtext = getOutput(gennet, transformer, img, offsetmidx, offsetmidy)

        offsetmidx += 64

        outtext2 = getOutput(gennet, transformer, img, offsetmidx, offsetmidy)

        offsetmidx -= 64

        nsize = 64
        strtx = offsetmidx - (nsize / 2)
        strty = offsetmidy - (nsize / 2)
        endx = offsetmidx + (nsize / 2)
        endy = offsetmidy + (nsize / 2)
        preimg = img[strtx:endx, strty:endy, 0]

        offsetmidx += 64

        nsize = 64
        strtx = offsetmidx - (nsize / 2)
        strty = offsetmidy - (nsize / 2)
        endx = offsetmidx + (nsize / 2)
        endy = offsetmidy + (nsize / 2)
        preimg2 = img[strtx:endx, strty:endy, 0]

        #outtext = (outtext - np.min(outtext)) / (np.max(outtext) - np.min(outtext))
        #outtext2 = (outtext2 - np.min(outtext2)) / (np.max(outtext2) - np.min(outtext2))
        #preimg = (preimg - np.min(preimg)) / (np.max(preimg) - np.min(preimg))

        io.imshow(np.concatenate((np.concatenate((outtext, preimg), axis=1), np.concatenate((outtext2, preimg2), axis=1)), axis=0))
        io.show()

        offsetmidx = 100
        offsetmidy = 120

        #outtext, preimg = getFractOutput(gennet, transformer, offsetmidx, offsetmidy)
        outtext, preimg = getLowOutput(gennet, transformer, lowgennet, offsetmidx, offsetmidy)

        offsetmidx += 64

        #outtext2, preimg2 = getFractOutput(gennet, transformer, offsetmidx, offsetmidy)
        outtext2, preimg2 = getLowOutput(gennet, transformer, lowgennet, offsetmidx, offsetmidy)

        #io.imshow(np.concatenate((outtext, outtext2), axis=0))
        io.imshow(np.concatenate((np.concatenate((outtext, preimg), axis=1), np.concatenate((outtext2, preimg2), axis=1)), axis=0))
        io.show()

    print "make picture"

    outimg = np.zeros((10*64,10*64), dtype=np.float)
    for offx in range(0,10):
        for offy in range(0,10):
            offsetmidx = offx*64 + 23820
            offsetmidy = offy*64 - 92039
            outimg[(offx*64):(offx*64+64),(offy*64):(offy*64+64)], _  = getLowOutput(gennet, transformer, lowgennet, offsetmidx, offsetmidy)

    #output binary

    #for extern usage, for example in the procedural terrain project
    f = open('HMaps\\Binaries\\HMap.bin', 'wb')
    data = struct.pack('<I', 10*64)
    data += struct.pack('<I', 10*64)
    data += struct.pack('<%sf' % (10*64*10*64), *outimg.flatten('F'))
    f.write(data)
    f.close()

    #output picture
    outimg = (outimg - np.min(outimg))/(np.max(outimg) - np.min(outimg))
    outimg *= 255.
    io.imshow(outimg)
    io.show()
    outimg = outimg.astype(np.uint8)

    io.imsave('HMaps\\HMap.png', outimg)

if __name__ == "__main__":
    main()