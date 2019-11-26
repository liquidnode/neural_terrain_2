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

import caffe
from caffe import layers as L
from caffe import params as P


upsample = True
inmemory = False

#params=[1e6*0.2, 1e6*0.2, 1e6*0.2, 0, 1e6*0.2, 0, 0, 1e6*0.2, 0, 0, 1e6*0.2, 0, 0]
params=[1e6*0.2, 0, 1e6*0.2, 0, 1e6*0.2, 0, 0, 1e6*0.2, 0, 0, 1e6*0.2, 0, 0]
#params=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    

def makeGram(n, i, shape, names, batchsize):
    if params[i] != 0:
        n['gram'+str(i+1)] = L.Python(n['relu'+names[i]], python_param=dict(module='gramlayer', 
                                                layer='GramLayer'), ntop=1)
        n['gram'+str(i+1)+'_s'] = L.Input(shape=shape,ntop=1)
        n['choosegram'+str(i+1)+'_s'] = L.Python(n['gram'+str(i+1)+'_s'], python_param=dict(module='chooselayer', 
                                                layer='ChooseData', param_str=str(batchsize)), ntop=1)

        n['loss_g'+str(i+1)] = L.EuclideanLoss(n['gram'+str(i+1)], n['choosegram'+str(i+1)+'_s'], loss_weight=params[i])



def initVGG(shps, w, names, n, input, batchsize, numgrams):
    maxl = 11

    n[input] = L.Reshape(n[input], name="strt", reshape_param={'shape':{'dim': [-1, 1, w, w]}})
    arglist = [n[input],n[input],n[input]]
    n.ddat = L.Concat(*arglist,concat_param={'axis':1})

    poolings = [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]

    convolution_param = dict(num_output=shps[0], kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n['conv'+names[0]] = L.Convolution(n.ddat, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
    n['relu'+names[0]] = L.ReLU(n['conv'+names[0]])

    for i in range(1,maxl):
        convolution_param = dict(num_output=shps[i], kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
        if poolings[i-1] == 1:
            n['conv'+names[i]] = L.Convolution(n['pool'+names[i-1]], param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
        else:
            n['conv'+names[i]] = L.Convolution(n['relu'+names[i-1]], param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
        n['relu'+names[i]] = L.ReLU(n['conv'+names[i]])
        if poolings[i] == 1 and i != maxl - 1:
            n['pool'+names[i]] = L.Pooling(n['relu'+names[i]], kernel_size=2, stride=2, pool=P.Pooling.MAX)

    for i in range(0,maxl):
        makeGram(n, i, [dict(dim=[numgrams, 1, shps[i], shps[i]])], names, batchsize)

    return n


def convBlock(name, opn, n, input, train=True):
    trainparam = []
    if train:
        trainparam = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
    else:
        trainparam = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]

    convolution_param = dict(num_output=opn, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n['gconv'+name+"_1"] = L.Convolution(n[input], convolution_param=convolution_param, param=trainparam)
    #if train:
    global_stat=True
    n['gconv'+name+"_1"] = L.BatchNorm(n['gconv'+name+"_1"], use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_1"] = L.ReLU(n['gconv'+name+"_1"], negative_slope=0.1)
    n['gconv'+name+"_2"] = L.Convolution(n['gelu'+name+"_1"], convolution_param=convolution_param, param=trainparam)
    #if train:
    n['gconv'+name+"_2"] = L.BatchNorm(n['gconv'+name+"_2"], use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_2"] = L.ReLU(n['gconv'+name+"_2"], negative_slope=0.1)
    convolution_param = dict(num_output=opn, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n['gconv'+name+"_3"] = L.Convolution(n['gelu'+name+"_2"], convolution_param=convolution_param, param=trainparam)
    #if train:
    n['gconv'+name+"_3"] = L.BatchNorm(n['gconv'+name+"_3"], use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_3"] = L.ReLU(n['gconv'+name+'_3'], negative_slope=0.1)
    return n, 'gelu'+name+'_3'

def joinBlock(name, opn, n, inputA, inputB, train=True):
    trainparam = []
    if train:
        trainparam = [dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 1)]
    else:
        trainparam = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]

     #TRAINABLE???? TODO
    if upsample == False:
        convolution_param = dict(num_output=opn, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
        n["upsample"+name] = L.Deconvolution(n[inputA], convolution_param=convolution_param, param=trainparam) 
    else:
        convolution_param = dict(num_output=opn, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='constant', value=0.0), bias_term=False)
        n["upsample"+name] = L.Deconvolution(n[inputA], convolution_param=convolution_param
                             ,param=[dict(lr_mult=0, decay_mult= 0)])
    #if train:
    global_stat=True

    n["upsampleB"+name] = L.BatchNorm(n["upsample"+name], use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n[inputB] = L.BatchNorm(n[inputB], use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    #ACHTUNG BEI BATCHNORM upsampleB genau da drunter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    n["concat"+name] = L.Concat(n["upsampleB"+name], n[inputB], concat_param=dict(axis=1))
    #else:
        #n["concat"+name] = L.Concat(n["upsample"+name], n[inputB], concat_param=dict(axis=1))
    return n, "concat"+name

def initGen(w, batchsize, numgrams, shps):
    names = ['1_1','1_2','2_1','2_2','3_1','3_2','3_3','4_1','4_2','4_3','5_1','5_2','5_3']

    n = caffe.NetSpec()
    if inmemory:
        n.alldata = L.Input(shape=[dict(dim=[16, 1, w, w])],ntop=1)
        n.data = L.Python(n.alldata, python_param=dict(module='chooselayer', layer='ChooseData', param_str=str(batchsize)), ntop=1)
    else:
        transform_param = dict(mirror=False, crop_size=w, scale=1., mean_value=103.939)
        n.data, n.nothing = L.ImageData(transform_param=transform_param, source='datasource.txt', 
                             is_color=False, shuffle=True, batch_size=batchsize, ntop=2)

    n["data0"] = L.Python(n.data, python_param=dict(module='randomblur', 
                                                   layer='RandomBlurData',
                                                   param_str='12'))
    codings = [8, 16, 24, 32, 40]
    #codings = [8, 8, 16, 28, 40]


    if False:
        #n.noth = L.Python(n.nothing, python_param=dict(module='destroy', layer='DestroyLayer'))
    
        n["blurdat"], n.lables = L.Python(n.data, python_param=dict(module='randomblur', 
                                                       layer='RandomBlurData',
                                                       param_str='12'), ntop=2)

        n = initVGG(shps, w, names, n, "blurdat")
        transf =  [8, 8, 8, 16, 16, 16, 28, 28, 28, 40, 40, 40, 60]
        pooling = [0, 0, 1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0]
        graming = [1, 0, 1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1]
        names2 = ['1_1','1_2','1_3','2_1','2_2','2_3','3_1','3_2','3_3','4_1','4_2','4_3','5_1']


        convolution_param = dict(num_output=transf[0], kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
        n['conv2'+names[0]] = L.Convolution(n["blurdat"], param=[dict(lr_mult=1, decay_mult= 1),dict(lr_mult=1, decay_mult= 0)], convolution_param=convolution_param)
        n['elu2'+names[0]] = L.ELU(n['conv2'+names[0]])

    d = w
    level = 5
    outname = ""
    for i in range(0,level):
        n["Zrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, d, d])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)#type='gaussian', std=127.5), ntop=1)#
        n["cdata"+str(i)] = L.Concat(n["Zrand_"+str(i)], n["data"+str(i)], concat_param=dict(axis=1))
        n, outname = convBlock("convA"+str(i), codings[0], n, "cdata"+str(i))
        if i != level-1:
            n["data"+str(i+1)]= L.Python(n["data"+str(i)], python_param=dict(module='halfsize', 
                                                           layer='HalfData'))
        d/=2


    #start joining
    n, outname = joinBlock("joinA", codings[0], n, outname, 'gelu'+'convA3'+'_3') # oder doch 8???

    n, outname = convBlock("convB", codings[1], n, outname)
    n, outname = joinBlock("joinB", codings[1], n, outname, 'gelu'+'convA2'+'_3')

    n, outname = convBlock("convC", codings[2], n, outname)
    n, outname = joinBlock("joinC", codings[2], n, outname, 'gelu'+'convA1'+'_3')

    n, outname = convBlock("convD", codings[3], n, outname)
    n, outname = joinBlock("joinD", codings[3], n, outname, 'gelu'+'convA0'+'_3')

    n, outname = convBlock("convE", codings[4], n, outname)
    convolution_param = dict(num_output=1, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n["texture"] = L.Convolution(n[outname], convolution_param=convolution_param)

    n.loss = L.EuclideanLoss(n["texture"], n.data, loss_weight=4e-4)
    #n.loss = L.EuclideanLoss(n["texture"], n["data0"], loss_weight=1e-6)

    n = initVGG(shps, w, names, n, "texture", batchsize, numgrams)

    return n.to_proto();


def initGAN(w, batchsize):
    n = caffe.NetSpec()
    if inmemory:
        n.alldata = L.Input(shape=[dict(dim=[16, 1, w, w])],ntop=1)
        n.data = L.Python(n.alldata, python_param=dict(module='chooselayer', layer='ChooseData', param_str=str(batchsize)), ntop=1)
    else:
        transform_param = dict(mirror=False, crop_size=w, scale=1., mean_value=103.939)
        n.data, n.nothing = L.ImageData(transform_param=transform_param, source='datasource.txt', 
                             is_color=False, shuffle=True, batch_size=batchsize, ntop=2)

    n["data0"] = L.Python(n.data, python_param=dict(module='blurdata', 
                                                   layer='BlurData',
                                                   param_str='12'))
    codings = [8, 16, 24, 32, 40]
    #codings = [8, 8, 16, 28, 40]
    

    d = w
    level = 5
    outname = ""
    for i in range(0,level):
        n["Zrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, d, d])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)#type='gaussian', std=127.5), ntop=1)#
        n["cdata"+str(i)] = L.Concat(n["Zrand_"+str(i)], n["data"+str(i)], concat_param=dict(axis=1))
        n, outname = convBlock("convA"+str(i), codings[0], n, "cdata"+str(i), train=False)
        if i != level-1:
            n["data"+str(i+1)]= L.Python(n["data"+str(i)], python_param=dict(module='halfsize', 
                                                           layer='HalfData'))
        d/=2


    #start joining
    n, outname = joinBlock("joinA", codings[0], n, outname, 'gelu'+'convA3'+'_3', train=False) # oder doch 8???

    n, outname = convBlock("convB", codings[1], n, outname, train=False)
    n, outname = joinBlock("joinB", codings[1], n, outname, 'gelu'+'convA2'+'_3', train=False)

    n, outname = convBlock("convC", codings[2], n, outname, train=False)
    n, outname = joinBlock("joinC", codings[2], n, outname, 'gelu'+'convA1'+'_3', train=False)

    n, outname = convBlock("convD", codings[3], n, outname, train=False)
    n, outname = joinBlock("joinD", codings[3], n, outname, 'gelu'+'convA0'+'_3', train=False)

    n, outname = convBlock("convE", codings[4], n, outname, train=False)
    convolution_param = dict(num_output=1, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n["texture"] = L.Convolution(n[outname], convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])

    #0 => blurdat
    #1 => data
    n.swap, n.labels = L.Python(n["texture"], n.data, python_param=dict(module='swaplayer', layer='SwapLayer'), propagate_down=[False, False], ntop=2)

    n.noise = L.DummyData(shape=[dict(dim=[batchsize, 1, w, w])], data_filler=dict(type='gaussian', std=20.), ntop=1)

    n.inp = L.Eltwise(n.swap, n.noise, eltwise_param={'operation':1})

    #GAN network
    global_stat = False
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganconv1 = L.Convolution(n.inp, convolution_param=convolution_param, name='convl1')
    n.ganconv1 = L.BatchNorm(n.ganconv1, use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganconv1 = L.ReLU(n.ganconv1, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganconv2 = L.Convolution(n.ganconv1, convolution_param=convolution_param, name='convl2')
    n.ganconv2 = L.BatchNorm(n.ganconv2, use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganconv2 = L.ReLU(n.ganconv2, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=2, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganconv3 = L.Convolution(n.ganconv2, convolution_param=convolution_param, name='convl3')
    n.ganconv3 = L.BatchNorm(n.ganconv3, use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganconv3 = L.ReLU(n.ganconv3, negative_slope=0.1)
    convolution_param = dict(num_output=8, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganconv4 = L.Convolution(n.ganconv3, convolution_param=convolution_param, name='convl4')
    n.ganconv4 = L.BatchNorm(n.ganconv4, use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganconv4 = L.ReLU(n.ganconv4, negative_slope=0.1)
    convolution_param = dict(num_output=8, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganconv5 = L.Convolution(n.ganconv4, convolution_param=convolution_param, name='convl5')
    n.ganconv5 = L.BatchNorm(n.ganconv5, use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganconv5 = L.ReLU(n.ganconv5, negative_slope=0.1)
    convolution_param = dict(num_output=4, kernel_size=2, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganconv6 = L.Convolution(n.ganconv5, convolution_param=convolution_param, name='convl6')
    n.ganconv6 = L.BatchNorm(n.ganconv6, use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganconv6 = L.ReLU(n.ganconv6, negative_slope=0.1)
    n.gangram = L.Python(n.ganconv6, python_param=dict(module='gramlayer',layer='GramLayer'), ntop=1)
    n.gangram = L.BatchNorm(n.gangram, use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ip1 = L.InnerProduct(n.gangram, num_output=8, weight_filler=dict(type='xavier'), name='full1')
    n.ip1 = L.BatchNorm(n.ip1, use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ip1 = L.ReLU(n.ip1, negative_slope=0.1)
    #n.ip1 = L.Dropout(n.ip1, dropout_param={"dropout_ratio":0.5})
    n.ip2 = L.InnerProduct(n.ip1, num_output=8, weight_filler=dict(type='xavier'), name='full2')
    n.ip2 = L.BatchNorm(n.ip2, use_global_stats=global_stat)#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ip2 = L.ReLU(n.ip2, negative_slope=0.1)
    #n.ip2 = L.Dropout(n.ip2, dropout_param={"dropout_ratio":0.5})
    n.ip3 = L.InnerProduct(n.ip2, num_output=2, weight_filler=dict(type='xavier'), name='full3')
    #n.prop = L.Softmax(n.ip3)
    n.loss = L.SoftmaxWithLoss(n.ip3, n.labels)


    return n.to_proto();

def initGANTest(w):

    n = caffe.NetSpec()

    n.ddata, n.labels = L.Python(python_param=dict(module='owndummy', 
                                            layer='ODummyData',
                                            param_str='1'), ntop=2)
    n.noth = L.Python(n.labels, python_param=dict(module='destroy', layer='DestroyLayer'))

    n.images = L.InnerProduct(n.ddata, num_output = w*w, weight_filler = dict(type='uniform', min=-114.8, max=255.0-114.8), bias_term=False)
    #n.images2 = L.InnerProduct(n.ddata, num_output = w*w, weight_filler = dict(type='uniform', min=-114.8, max=255.0-114.8), bias_term=False)
    #n.images3 = L.InnerProduct(n.ddata, num_output = w*w, weight_filler = dict(type='uniform', min=-114.8, max=255.0-114.8), bias_term=False)
    #n.images4 = L.InnerProduct(n.ddata, num_output = w*w, weight_filler = dict(type='uniform', min=-114.8, max=255.0-114.8), bias_term=False)
    #n.images5 = L.InnerProduct(n.ddata, num_output = w*w, weight_filler = dict(type='uniform', min=-114.8, max=255.0-114.8), bias_term=False)
    #n.images6 = L.InnerProduct(n.ddata, num_output = w*w, weight_filler = dict(type='uniform', min=-114.8, max=255.0-114.8), bias_term=False)
    #n.images7 = L.InnerProduct(n.ddata, num_output = w*w, weight_filler = dict(type='uniform', min=-114.8, max=255.0-114.8), bias_term=False)
    #n.images8 = L.InnerProduct(n.ddata, num_output = w*w, weight_filler = dict(type='uniform', min=-114.8, max=255.0-114.8), bias_term=False)
    #n.conc = L.Concat(n.images, n.images2, n.images3, n.images4, n.images5, n.images6, n.images7, n.images8, axis=0)
    n.smaller = L.Reshape(n.images, reshape_param={'shape':{'dim': [1, 1, w, w]}})


    if False:
        convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
        n.ganconv1 = L.Convolution(n.smaller, convolution_param=convolution_param, name='convl1', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
        n.ganconv1 = L.ELU(n.ganconv1)
        convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
        n.ganconv2 = L.Convolution(n.ganconv1, convolution_param=convolution_param, name='convl2', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
        n.ganconv2 = L.ELU(n.ganconv2)
        convolution_param = dict(num_output=32, kernel_size=2, stride=1, pad=0, weight_filler = dict(type='xavier'))
        n.ganconv3 = L.Convolution(n.ganconv2, convolution_param=convolution_param, name='convl3', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
        n.ganconv3 = L.ELU(n.ganconv3)
        convolution_param = dict(num_output=24, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
        n.ganconv4 = L.Convolution(n.ganconv3, convolution_param=convolution_param, name='convl4', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
        n.ganconv4 = L.ELU(n.ganconv4)
        convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
        n.ganconv5 = L.Convolution(n.ganconv4, convolution_param=convolution_param, name='convl5', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
        n.ganconv5 = L.ELU(n.ganconv5)
        convolution_param = dict(num_output=16, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
        n.ganconv6 = L.Convolution(n.ganconv5, convolution_param=convolution_param, name='convl6', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
        n.ganconv6 = L.ELU(n.ganconv6)
        n.gangram = L.Python(n.ganconv6, python_param=dict(module='gramlayer',layer='GramLayer'), ntop=1)
        n.ip1 = L.InnerProduct(n.gangram, num_output=128, weight_filler=dict(type='xavier'), name='full1', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
        n.ip1 = L.ELU(n.ip1)
        n.ip2 = L.InnerProduct(n.ip1, num_output=128, weight_filler=dict(type='xavier'), name='full2', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
        n.ip2 = L.ELU(n.ip2)
        n.ip3 = L.InnerProduct(n.ip2, num_output=2, weight_filler=dict(type='xavier'), name='full3', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
       
        
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganconv1 = L.Convolution(n.smaller, convolution_param=convolution_param, name='convl1', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
    n.ganconv1 = L.BatchNorm(n.ganconv1, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}], use_global_stats=True)
    n.ganconv1 = L.ReLU(n.ganconv1, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganconv2 = L.Convolution(n.ganconv1, convolution_param=convolution_param, name='convl2', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
    n.ganconv2 = L.BatchNorm(n.ganconv2, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}], use_global_stats=True)
    n.ganconv2 = L.ReLU(n.ganconv2, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=2, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganconv3 = L.Convolution(n.ganconv2, convolution_param=convolution_param, name='convl3', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
    n.ganconv3 = L.BatchNorm(n.ganconv3, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}], use_global_stats=True)
    n.ganconv3 = L.ReLU(n.ganconv3, negative_slope=0.1)
    convolution_param = dict(num_output=8, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganconv4 = L.Convolution(n.ganconv3, convolution_param=convolution_param, name='convl4', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
    n.ganconv4 = L.BatchNorm(n.ganconv4, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}], use_global_stats=True)
    n.ganconv4 = L.ReLU(n.ganconv4, negative_slope=0.1)
    convolution_param = dict(num_output=8, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganconv5 = L.Convolution(n.ganconv4, convolution_param=convolution_param, name='convl5', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
    n.ganconv5 = L.BatchNorm(n.ganconv5, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}], use_global_stats=True)
    n.ganconv5 = L.ReLU(n.ganconv5, negative_slope=0.1)
    convolution_param = dict(num_output=4, kernel_size=2, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganconv6 = L.Convolution(n.ganconv5, convolution_param=convolution_param, name='convl6', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
    n.ganconv6 = L.BatchNorm(n.ganconv6, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}], use_global_stats=True)
    n.ganconv6 = L.ReLU(n.ganconv6, negative_slope=0.1)
    n.gangram = L.Python(n.ganconv6, python_param=dict(module='gramlayer',layer='GramLayer'), ntop=1)
    n.gangram = L.BatchNorm(n.gangram, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}], use_global_stats=True)
    n.ip1 = L.InnerProduct(n.gangram, num_output=8, weight_filler=dict(type='xavier'), name='full1', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
    n.ip1 = L.BatchNorm(n.ip1, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}], use_global_stats=True)
    n.ip1 = L.ReLU(n.ip1, negative_slope=0.1)
    n.ip2 = L.InnerProduct(n.ip1, num_output=8, weight_filler=dict(type='xavier'), name='full2', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
    n.ip2 = L.BatchNorm(n.ip2, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}], use_global_stats=True)
    n.ip2 = L.ReLU(n.ip2, negative_slope=0.1)
    n.ip3 = L.InnerProduct(n.ip2, num_output=2, weight_filler=dict(type='xavier'), name='full3', param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)])
         
    n.loss = L.SoftmaxWithLoss(n.ip3, n.noth, propagate_down=[True, False])

    return n.to_proto();


def main():
    pycaffe_dir = os.path.dirname(__file__)

#    caffe.set_random_seed(10923)
    caffe.set_mode_gpu();
    w = 96
    maxl = 11
    batchsize = 4
    numgrams = 4
    shps = [64,64,128,128,256,256,256,512,512,512,512,512,512]

    with open('teraGenNet.prototxt', 'w') as f:
        n = initGen(w, batchsize, numgrams, shps)
        f.write(str(n))

    solver = caffe.get_solver('solver_3.prototxt')
    solver.restore('teraGenNet2_adadelta_iter_22000.solverstate')

    if False:
        vggnet = caffe.Net('VGG.prototxt','VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)

        for key in solver.net.params.keys():
            if key != 'images' and key in vggnet.params:
                solver.net.params[key][0].data[...] = vggnet.params[key][0].data
                solver.net.params[key][1].data[...] = vggnet.params[key][1].data

    #print solver.net.params["upsample"+"joinB"][0].data.shape
    #ACHTUNG IST IMMER TRUE !!!! WENN NICHT RESTORED
    if False:#upsample:
        for i in range(0,8):
            solver.net.params["upsample"+"joinA"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
        for i in range(0,16):
            solver.net.params["upsample"+"joinB"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
        for i in range(0,24):
            solver.net.params["upsample"+"joinC"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
        for i in range(0,32):
            solver.net.params["upsample"+"joinD"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])

    img = caffe.io.load_image('newdata/test_data_3500_1000.bmp', color=False)
    transformer = caffe.io.Transformer({'data': (1,1,w,w)})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 1.0)
    #transformer.set_mean('data',np.array([114.8],np.float32))
    width = img.shape[0]
    heigth = img.shape[1]
    #img-=114.8/255.0
    img*=255.0
    img-=103.939#/255.0
    #img-=114.8
    print width
    print heigth


    blbs = []
    all_g = []
    for i in range(1,maxl+1):
        if params[i-1] != 0:
            blbs.append('gram'+str(i))
            all_g.append(np.zeros((numgrams, 1, shps[i-1], shps[i-1]), dtype=np.float32))

    print blbs 


    #offlistx = [25, 0, 96, 200]
    #offlisty = [15, 100, 0, 30]
    offlistx = [25, 25, 25, 25]
    offlisty = [15, 15, 15, 15]
    
    for i in range(0,numgrams/batchsize):
        for j in range(0, batchsize):
            offsetx = offlistx[j+i*batchsize]#randint(0, width - (w + 1)) 
            offsety = offlisty[j+i*batchsize]#randint(0, heigth - (w + 1)) 
            deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]
            #deimg = ((deimg - np.min(deimg))/(np.max(deimg) - np.min(deimg)))*255.-103.939
            #io.imshow(deimg.reshape(w,w))
            #io.show()
            solver.net.blobs['texture'].data[j,:,:,:] = transformer.preprocess('data', deimg)
        outp = solver.net.forward(blobs=blbs, start='strt', end='loss_g'+str(11))

        #io.imshow(np.concatenate((255*deimg.reshape((w,w)), 255*solver.net.blobs['relu1_1'].data[0,0,:,:].reshape((w,w))), axis=1))
        #io.show()
        for k in range(0,len(blbs)):
            all_g[k][(i*batchsize):((i+1)*batchsize):1,:,:,:] = outp[blbs[k]]

    
    if inmemory:
        for j in range(0, 16):
            offsetx = randint(0, width - (w + 1)) 
            offsety = randint(0, heigth - (w + 1)) 
            deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]
            #deimg = ((deimg - np.min(deimg))/(np.max(deimg) - np.min(deimg)))*255.-103.939
            solver.net.blobs['alldata'].data[j,:,:,:] = transformer.preprocess('data', deimg)
    
    #print all_g[9]
    #for i in range(0,maxl):
    #    all_g[i] *= 1.0 / float(numiter)

    offsetx=25
    offsety=15
    deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]

    for i in range(0,len(blbs)):
        solver.net.blobs[blbs[i]+'_s'].data[...] = all_g[i]


    for i in range (0,15):
        solver.step(200)
        deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]
        solver.net.blobs['data0'].data[0,:,:,:] = transformer.preprocess('data', cv2.blur(deimg.reshape(w,w),(12, 12)).reshape(w,w,1))
        outp = solver.net.forward(blobs=['texture'], start="Zrand_"+str(0))

        dat = np.array(outp['texture'][0,:,:,:], copy=True)  

        solver.net.blobs['data0'].data[0,:,:,:] = transformer.preprocess('data', cv2.blur(dat.reshape(w,w),(12, 12)).reshape(w,w,1))
        outp = solver.net.forward(blobs=['texture'], start="Zrand_"+str(0))

        dat2 = np.array(outp['texture'][0,:,:,:], copy=True)  

        solver.net.blobs['data0'].data[0,:,:,:] = transformer.preprocess('data', cv2.blur(dat2.reshape(w,w),(12, 12)).reshape(w,w,1))
        outp = solver.net.forward(blobs=['texture'], start="Zrand_"+str(0))

        dat3 = np.array(outp['texture'][0,:,:,:], copy=True)  

        io.imshow(2.0*np.concatenate((deimg.reshape((w,w)), cv2.blur(deimg.reshape(w,w),(12, 12)), np.reshape(dat ,(w,w)), np.reshape(dat2 ,(w,w)), np.reshape(dat3 ,(w,w))), axis=1))
        io.show()

def main2():
    pycaffe_dir = os.path.dirname(__file__)

#    caffe.set_random_seed(10923)
    caffe.set_mode_gpu();
    w = 96
    batchsize = 8



    with open('teraGAN.prototxt', 'w') as f:
        n = initGAN(w, batchsize)
        f.write(str(n))


    orig = caffe.get_solver('solver_3.prototxt')
    orig.restore('teraGenNet2_adadelta_iter_20200.solverstate')
    solver = caffe.get_solver('solver_4.prototxt')
    solver.restore('teraGAN_adadelta_iter_1000.solverstate')
    #solver.restore('teraGAN_adadelta_iter_103000.solverstate')

    for key in solver.net.params.keys():
        if key in orig.net.params:
            solver.net.params[key][0].data[...] = orig.net.params[key][0].data
            if len(solver.net.params[key]) == 2:
                solver.net.params[key][1].data[...] = orig.net.params[key][1].data
    
    solver.net.params["texture"][0].data[...] = orig.net.params["Convolution28"][0].data
    solver.net.params["texture"][1].data[...] = orig.net.params["Convolution28"][1].data

    #for k in solver.net.params.keys():
    #    print "\"" + k + "\","

    img = caffe.io.load_image('newdata/test_data_3500_1000.bmp', color=False)
    transformer = caffe.io.Transformer({'data': (1,1,w,w)})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 1.0)
    #transformer.set_mean('data',np.array([114.8],np.float32))
    width = img.shape[0]
    heigth = img.shape[1]
    #img-=114.8/255.0
    img*=255.0
    img-=103.939#/255.0
    #img-=114.8
    print width
    print heigth

    if inmemory:
        for j in range(0, 16):
            offsetx = randint(0, width - (w + 1)) 
            offsety = randint(0, heigth - (w + 1)) 
            deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]
            #deimg = ((deimg - np.min(deimg))/(np.max(deimg) - np.min(deimg)))*255.-103.939
            solver.net.blobs['alldata'].data[j,:,:,:] = transformer.preprocess('data', deimg)
    
    solver.step(1000)
    outp = solver.net.forward(blobs=["ip3", "labels"])
    print outp["labels"]
    float_formatter = lambda x: "%.2f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print np.exp(outp["ip3"][:,0]) / (np.exp(outp["ip3"][:,0]) + np.exp(outp["ip3"][:,1]))
    np.set_printoptions()

    io.imshow(2.0*np.concatenate((solver.net.blobs["swap"].data[0,:,:,:].reshape((w,w)), solver.net.blobs["swap"].data[1,:,:,:].reshape(w,w), np.reshape(solver.net.blobs["swap"].data[2,:,:,:] ,(w,w))), axis=1))
    io.show()

    offsetx=25
    offsety=15
    deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]
    dat = solver.net.blobs["texture"].data[0,:,:,:]
    #dat = ((dat - np.min(dat))/(np.max(dat) - np.min(dat)))*255.-103.939
    io.imshow(2.0*np.concatenate((solver.net.blobs["data"].data[0,:,:,:].reshape((w,w)), cv2.blur(solver.net.blobs["data"+str(0)].data[0,:,:,:].reshape(w,w),(12, 12)), np.reshape(dat ,(w,w))), axis=1))
    io.show()

def GANTest():
    pycaffe_dir = os.path.dirname(__file__)

#    caffe.set_random_seed(10923)
    caffe.set_mode_gpu();
    w = 96

    with open('teraGAN.prototxt', 'w') as f:
        n = initGAN(w, 8)
        f.write(str(n))
    with open('teraGANTest.prototxt', 'w') as f:
        n = initGANTest(w)
        f.write(str(n))

    orig = caffe.get_solver('solver_4.prototxt')
    orig.restore('teraGAN_adadelta_iter_1000.solverstate')

    if False:
        outp = orig.net.forward(blobs=["ip3", "labels"])
        print outp["labels"]
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})
        print np.exp(outp["ip3"][:,0]) / (np.exp(outp["ip3"][:,0]) + np.exp(outp["ip3"][:,1]))
        np.set_printoptions()


    solver = caffe.get_solver('solver_5.prototxt')

    for key in solver.net.params.keys():
        if key in orig.net.params:
            for i in range(0,len(solver.net.params[key])):
                solver.net.params[key][i].data[...] = orig.net.params[key][i].data

    img = caffe.io.load_image('newdata/test_data_3500_1000.bmp', color=False)
    transformer = caffe.io.Transformer({'data': (1,1,w,w)})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 1.0)
    #transformer.set_mean('data',np.array([114.8],np.float32))
    width = img.shape[0]
    heigth = img.shape[1]
    #img-=114.8/255.0
    img*=255.0
    img-=103.939#/255.0

    offsetx = 25
    offsety = 15 
    deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]

    if inmemory:
        for j in range(0, 16):
            offsetx = 25#randint(0, width - (w + 1)) 
            offsety = 15#randint(0, heigth - (w + 1)) 
            deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]
            #deimg = ((deimg - np.min(deimg))/(np.max(deimg) - np.min(deimg)))*255.-103.939
            orig.net.blobs['alldata'].data[j,:,:,:] = transformer.preprocess('data', deimg)

    outp = orig.net.forward(blobs=['texture','ip3','labels'])

    dat = np.array(outp['texture'][0,:,:,:], copy=True) 
    print outp["labels"]
    float_formatter = lambda x: "%.2f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print 1.0 / (1.0 + np.exp(outp["ip3"][:,1]-outp["ip3"][:,0]))
    np.set_printoptions()

    solver.net.params['images'][0].data[...] = np.reshape(transformer.preprocess('data', deimg.reshape(w,w,1)), (w*w,1))#+(np.random.rand(w*w,1)-0.5)*20.

    io.imshow(2.0*np.concatenate((deimg.reshape((w,w)), cv2.blur(deimg.reshape(w,w),(12, 12)), np.reshape(dat ,(w,w))), axis=1))
    io.show()

    print solver.net.forward()
    dat = deimg.reshape(1,1,w,w)
    #solver.net.params['images'][0].data[...] = np.reshape(transformer.preprocess('data', dat.reshape(w,w,1)), (w*w,1))

    print solver.net.forward()

    outp = solver.net.forward(blobs=["ip3"])
    float_formatter = lambda x: "%.2f" % x
    np.set_printoptions(formatter={'float_kind':float_formatter})
    print 1.0 / (1.0 + np.exp(outp["ip3"][:,1]-outp["ip3"][:,0]))
    np.set_printoptions()


    for i in range (0,15):
        solver.step(1000)
        dat2 = solver.net.blobs['smaller'].data
        float_formatter = lambda x: "%.2f" % x
        np.set_printoptions(formatter={'float_kind':float_formatter})
        print 1.0 / (1.0 + np.exp(outp["ip3"][:,1]-outp["ip3"][:,0]))
        np.set_printoptions()

        #dat = ((dat - np.min(dat))/(np.max(dat) - np.min(dat)))*255.-103.939
        #dat = (dat >= -1.0) * dat
        #dat = (dat <= 1.0) * dat
        datD = (dat-dat2)
        datD = ((datD - np.min(datD))/(np.max(datD) - np.min(datD)))*255.-103.939
        io.imshow(2.0*np.concatenate((dat.reshape((w,w)), datD.reshape(w,w), np.reshape(dat2 ,(w,w))), axis=1))
        io.show()

if __name__ == "__main__":
    #main()
    main2()
    #GANTest()

