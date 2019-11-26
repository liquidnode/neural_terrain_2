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

inmemory = False
upsample = True
global_stat = False
full_conv = True
finetune = False
original_model = False
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
    if global_stat:
        trainparam2 = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
    
    convolution_param = dict(num_output=opn, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    if full_conv:
        convolution_param = dict(num_output=opn, kernel_size=3, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n['gconv'+name+"_1"] = L.Convolution(n[input], convolution_param=convolution_param, param=trainparam, name=allparamnames.pop(0))
    #if train:
    n['gconv'+name+"_1"] = L.BatchNorm(n['gconv'+name+"_1"], use_global_stats=global_stat, param=trainparam2, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n['gelu'+name+"_1"] = L.ReLU(n['gconv'+name+"_1"], negative_slope=0.1)

    if finetune:
        trainparam = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
        trainparam2 = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]

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
    if global_stat:
        trainparam2 = [dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)]
    if upsample == False:
        convolution_param = dict(num_output=opn, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
        n["upsample"+name] = L.Deconvolution(n[inputA], convolution_param=convolution_param, param=trainparam, name=allparamnames.pop(0)) 
    else:
        convolution_param = dict(num_output=opn, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='constant', value=0.0), bias_term=False)
        n["upsample"+name] = L.Deconvolution(n[inputA], convolution_param=convolution_param
                             ,param=[dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))

    n["upsampleB"+name] = L.BatchNorm(n["upsample"+name], use_global_stats=global_stat, param=trainparam2, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n[inputB] = L.BatchNorm(n[inputB], use_global_stats=global_stat, param=trainparam2, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n["concat"+name] = L.Concat(n["upsampleB"+name], n[inputB], concat_param=dict(axis=1))
    return n, "concat"+name

def AGAN(w, batchsize, n):
    listofsizes =  []
    if full_conv:
        listofsizes = [w]
    
        level = 5
        for i in range(0, level-1):
            alast = listofsizes[i]
            listofsizes.append((alast - 4)*2)
        listofsizes[0] -= 4


    if inmemory:
        n.Adata = L.Python(n.alldata, python_param=dict(module='chooselayer', layer='ChooseData', param_str=str(batchsize)), ntop=1)
    else:
        transform_param = dict(mirror=False, crop_size=w, scale=1., mean_value=103.939)
        if full_conv:
            transform_param = dict(mirror=False, crop_size=listofsizes[level -1] - 4, scale=1., mean_value=103.939)
        n.Adata, n.Anothing = L.ImageData(transform_param=transform_param, source='datasource.txt', 
                             is_color=False, shuffle=True, batch_size=batchsize, ntop=2)

    codings = [8, 16, 24, 32, 40]
    #codings = [8, 8, 16, 28, 40]

    
    d = w
    level = 5
    outname = ""
    for i in range(0,level):
        if full_conv:
            n["AZrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, listofsizes[level-1 -i] + 4, listofsizes[level-1 -i] + 4])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)#type='gaussian', std=127.5), ntop=1)#
        else:
            n["AZrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, d, d])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)#type='gaussian', std=127.5), ntop=1)#
        n, outname = convBlock("AconvA"+str(i), codings[0], n, "AZrand_"+str(i), train=False)
        d /= 2


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

    #0 => blurdat
    #1 => data
    n.Aswap, n.Alabels = L.Python(n["Atexture"], n.Adata, python_param=dict(module='swaplayer', layer='SwapLayer'), propagate_down=[False, False], ntop=2)

    if full_conv:
        n.Anoise = L.DummyData(shape=[dict(dim=[batchsize, 1, listofsizes[level -1] - 4, listofsizes[level -1] - 4])], data_filler=dict(type='gaussian', std=2.0), ntop=1)
    else:
        n.Anoise = L.DummyData(shape=[dict(dim=[batchsize, 1, w, w])], data_filler=dict(type='gaussian', std=2.0), ntop=1)

    n.Ainp = L.Eltwise(n.Aswap, n.Anoise, eltwise_param={'operation':1})

    #GAN network
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv1 = L.Convolution(n.Ainp, convolution_param=convolution_param, name=allparamnames.pop(0))
    #n.ganAconv1 = L.BatchNorm(n.ganAconv1, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv1 = L.ReLU(n.ganAconv1, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv2 = L.Convolution(n.ganAconv1, convolution_param=convolution_param, name=allparamnames.pop(0))
    #n.ganAconv2 = L.BatchNorm(n.ganAconv2, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv2 = L.ReLU(n.ganAconv2, negative_slope=0.1)
    convolution_param = dict(num_output=32, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganAconv3 = L.Convolution(n.ganAconv2, convolution_param=convolution_param, name=allparamnames.pop(0))
    #n.ganAconv3 = L.BatchNorm(n.ganAconv3, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv3 = L.ReLU(n.ganAconv3, negative_slope=0.1)

    convolution_param = dict(num_output=32, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv4 = L.Convolution(n.ganAconv3, convolution_param=convolution_param, name=allparamnames.pop(0))
    #n.ganAconv4 = L.BatchNorm(n.ganAconv4, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv4 = L.ReLU(n.ganAconv4, negative_slope=0.1)
    convolution_param = dict(num_output=32, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv5 = L.Convolution(n.ganAconv4, convolution_param=convolution_param, name=allparamnames.pop(0))
    #n.ganAconv5 = L.BatchNorm(n.ganAconv5, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv5 = L.ReLU(n.ganAconv5, negative_slope=0.1)
    convolution_param = dict(num_output=32, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganAconv6 = L.Convolution(n.ganAconv5, convolution_param=convolution_param, name=allparamnames.pop(0))
    #n.ganAconv6 = L.BatchNorm(n.ganAconv6, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv6 = L.ReLU(n.ganAconv6, negative_slope=0.1)

    convolution_param = dict(num_output=32, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganAconv7 = L.Convolution(n.ganAconv6, convolution_param=convolution_param, name=allparamnames.pop(0))
    #n.ganAconv7 = L.BatchNorm(n.ganAconv7, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv7 = L.ReLU(n.ganAconv7, negative_slope=0.1)
    n.ganAconv7_pool = L.Pooling(n.ganAconv7, global_pooling=True, pool=P.Pooling.AVE)





    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv8 = L.Convolution(n.ganAconv6, convolution_param=convolution_param, name="addit1")
    #n.ganAconv7 = L.BatchNorm(n.ganAconv7, use_global_stats=global_stat, name="addit2")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv8 = L.ReLU(n.ganAconv8, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv9 = L.Convolution(n.ganAconv8, convolution_param=convolution_param, name="addit2")
    #n.ganAconv8 = L.BatchNorm(n.ganAconv8, use_global_stats=global_stat, name="addit4")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv9 = L.ReLU(n.ganAconv9, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganAconv10 = L.Convolution(n.ganAconv9, convolution_param=convolution_param, name="addit3")
    #n.ganAconv9 = L.BatchNorm(n.ganAconv9, use_global_stats=global_stat, name="addit6")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv10 = L.ReLU(n.ganAconv10, negative_slope=0.1)

    convolution_param = dict(num_output=16, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganAconv11 = L.Convolution(n.ganAconv10, convolution_param=convolution_param, name="addit4")
    #n.ganAconv7 = L.BatchNorm(n.ganAconv7, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv11 = L.ReLU(n.ganAconv11, negative_slope=0.1)
    if original_model:
        n.ganAconv11_pool = L.Pooling(n.ganAconv11, global_pooling=True, pool=P.Pooling.AVE)
    else:
        n.ganAip4 = L.InnerProduct(n.ganAconv11, num_output=32, weight_filler=dict(type='xavier'), name="ip4")
        n.ganAip4 = L.ReLU(n.ganAip4, negative_slope=0.1)
        n.ganAip4 = L.Dropout(n.ganAip4, dropout_param={"dropout_ratio":0.5})
        n.ganAip5 = L.InnerProduct(n.ganAip4, num_output=16, weight_filler=dict(type='xavier'), name="ip5")
        n.ganAip5 = L.ReLU(n.ganAip5, negative_slope=0.1)
        n.ganAip5_resh = L.Reshape(n.ganAip5, reshape_param={'shape':{'dim': [batchsize, -1, 1, 1]}})


    convolution_param = dict(num_output=16, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganAconv12 = L.Convolution(n.ganAconv3, convolution_param=convolution_param, name="addit5")
    #n.ganAconv7 = L.BatchNorm(n.ganAconv7, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv12 = L.ReLU(n.ganAconv12, negative_slope=0.1)
    n.ganAconv12_pool = L.Pooling(n.ganAconv12, global_pooling=True, pool=P.Pooling.AVE)

    if original_model:
        n.Aconc_gan = L.Concat(n.ganAconv7_pool, n.ganAconv11_pool, n.ganAconv12_pool, concat_param=dict(axis=1))
    else:
        n.Aconc_gan = L.Concat(n.ganAconv7_pool, n.ganAip5_resh, n.ganAconv12_pool, concat_param=dict(axis=1))


    n.Aip3 = L.InnerProduct(n.Aconc_gan, num_output=1, weight_filler=dict(type='xavier'), name="last")
    
    n.Aloss = L.SigmoidCrossEntropyLoss(n.Aip3, n.Alabels)

    return n


def ABGAN(w, batchsize, n):
    n.ABnothing = L.DummyData(shape=[dict(dim=[batchsize, 1, 1, 1])], data_filler=dict(type='constant'), ntop=1)
    n.ABlabels = L.Python(n.ABnothing, python_param=dict(module='destroy', layer='DestroyLayer'))
    codings = [8, 16, 24, 32, 40]
    #codings = [8, 8, 16, 28, 40]
    
    listofsizes =  []
    if full_conv:
        listofsizes = [w]
    
        level = 5
        for i in range(0, level-1):
            alast = listofsizes[i]
            listofsizes.append((alast - 4)*2)
        listofsizes[0] -= 4

    d = w
    level = 5
    outname = ""
    for i in range(0,level):
        if full_conv:
            n["ABZrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, listofsizes[level-1 -i] + 4, listofsizes[level-1 -i] + 4])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)
        else:
            n["ABZrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, d, d])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)
        n, outname = convBlock("ABconvA"+str(i), codings[0], n, "ABZrand_"+str(i), train=True)
        d /= 2


    #START True
    #start joining
    n, outname = joinBlock("ABjoinA", codings[0], n, outname, 'gelu'+'ABconvA3'+'_3', train=True) # oder doch 8???

    n, outname = convBlock("ABconvB", codings[1], n, outname, train=True)
    n, outname = joinBlock("ABjoinB", codings[1], n, outname, 'gelu'+'ABconvA2'+'_3', train=True)

    n, outname = convBlock("ABconvC", codings[2], n, outname, train=True)
    n, outname = joinBlock("ABjoinC", codings[2], n, outname, 'gelu'+'ABconvA1'+'_3', train=True)

    n, outname = convBlock("ABconvD", codings[3], n, outname, train=True)
    n, outname = joinBlock("ABjoinD", codings[3], n, outname, 'gelu'+'ABconvA0'+'_3', train=True)

    n, outname = convBlock("ABconvE", codings[4], n, outname, train=True)
    convolution_param = dict(num_output=1, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n["ABtexture"] = L.Convolution(n[outname], convolution_param=convolution_param, name=allparamnames.pop(0))

    #GAN network
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv1 = L.Convolution(n["ABtexture"], convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))
    #n.ganABconv1 = L.BatchNorm(n.ganABconv1, use_global_stats=global_stat, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv1 = L.ReLU(n.ganABconv1, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv2 = L.Convolution(n.ganABconv1, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))
    #n.ganABconv2 = L.BatchNorm(n.ganABconv2, use_global_stats=global_stat, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv2 = L.ReLU(n.ganABconv2, negative_slope=0.1)
    convolution_param = dict(num_output=32, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganABconv3 = L.Convolution(n.ganABconv2, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))
    #n.ganABconv3 = L.BatchNorm(n.ganABconv3, use_global_stats=global_stat, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv3 = L.ReLU(n.ganABconv3, negative_slope=0.1)

    convolution_param = dict(num_output=32, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv4 = L.Convolution(n.ganABconv3, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))
    #n.ganABconv4 = L.BatchNorm(n.ganABconv4, use_global_stats=global_stat, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv4 = L.ReLU(n.ganABconv4, negative_slope=0.1)
    convolution_param = dict(num_output=32, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv5 = L.Convolution(n.ganABconv4, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))
    #n.ganABconv5 = L.BatchNorm(n.ganABconv5, use_global_stats=global_stat, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv5 = L.ReLU(n.ganABconv5, negative_slope=0.1)
    convolution_param = dict(num_output=32, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganABconv6 = L.Convolution(n.ganABconv5, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))
    #n.ganABconv6 = L.BatchNorm(n.ganABconv6, use_global_stats=global_stat, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv6 = L.ReLU(n.ganABconv6, negative_slope=0.1)

    convolution_param = dict(num_output=32, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganABconv7 = L.Convolution(n.ganABconv6, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))
    #n.ganABconv7 = L.BatchNorm(n.ganABconv7, use_global_stats=global_stat, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv7 = L.ReLU(n.ganABconv7, negative_slope=0.1)
    n.ganABconv7_pool = L.Pooling(n.ganABconv7, global_pooling=True, pool=P.Pooling.AVE)



    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv8 = L.Convolution(n.ganABconv6, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="addit1")
    #n.ganABconv7 = L.BatchNorm(n.ganABconv7, use_global_stats=global_stat, name="addit2")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv8 = L.ReLU(n.ganABconv8, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv9 = L.Convolution(n.ganABconv8, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="addit2")
    #n.ganABconv8 = L.BatchNorm(n.ganABconv8, use_global_stats=global_stat, name="addit4")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv9 = L.ReLU(n.ganABconv9, negative_slope=0.1)
    convolution_param = dict(num_output=16, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganABconv10 = L.Convolution(n.ganABconv9, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="addit3")
    #n.ganABconv9 = L.BatchNorm(n.ganABconv9, use_global_stats=global_stat, name="addit6")#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv10 = L.ReLU(n.ganABconv10, negative_slope=0.1)

    convolution_param = dict(num_output=16, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganABconv11 = L.Convolution(n.ganABconv10, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="addit4")
    #n.ganABconv7 = L.BatchNorm(n.ganABconv7, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv11 = L.ReLU(n.ganABconv11, negative_slope=0.1)

    if original_model:
        n.ganABconv11_pool = L.Pooling(n.ganABconv11, global_pooling=True, pool=P.Pooling.AVE)
    else:
        n.ganABip4 = L.InnerProduct(n.ganABconv11, num_output=32, weight_filler=dict(type='xavier'), param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="ip4")
        n.ganABip4 = L.ReLU(n.ganABip4, negative_slope=0.1)
        #n.ganABip4 = L.Dropout(n.ganABip4, dropout_param={"dropout_ratio":0.5})
        n.ganABip5 = L.InnerProduct(n.ganABip4, num_output=16, weight_filler=dict(type='xavier'), param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="ip5")
        n.ganABip5 = L.ReLU(n.ganABip5, negative_slope=0.1)
        n.ganABip5_resh = L.Reshape(n.ganABip5, reshape_param={'shape':{'dim': [batchsize, -1, 1, 1]}})

    convolution_param = dict(num_output=16, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganABconv12 = L.Convolution(n.ganABconv3, convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="addit5")
    #n.ganABconv7 = L.BatchNorm(n.ganABconv7, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv12 = L.ReLU(n.ganABconv12, negative_slope=0.1)
    n.ganABconv12_pool = L.Pooling(n.ganABconv12, global_pooling=True, pool=P.Pooling.AVE)

    if original_model:
        n.ABconc_gan = L.Concat(n.ganABconv7_pool, n.ganABconv11_pool, n.ganABconv12_pool, concat_param=dict(axis=1))
    else:
        n.ABconc_gan = L.Concat(n.ganABconv7_pool, n.ganABip5_resh, n.ganABconv12_pool, concat_param=dict(axis=1))


    n.ABip3 = L.InnerProduct(n.ABconc_gan, num_output=1, weight_filler=dict(type='xavier'), param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="last")
    
    n.ABloss = L.SigmoidCrossEntropyLoss(n.ABip3, n.ABlabels)

    return n


def initOGAN(w, batchsize, p=0):
    global allparamnames
    n = caffe.NetSpec()
    if inmemory:
        n.alldata = L.Input(shape=[dict(dim=[16, 1, w, w])],ntop=1)
    if p == 0:
        n = ABGAN(w, batchsize, n)
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

        n = AGAN(w, batchsize, n)
    else:
        if p == 1:
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

            n = AGAN(w, batchsize, n)
        else:
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
            n = ABGAN(w, batchsize, n)
    return n.to_proto()

def copyParams(orig, solver):
    #print "start copy"
    for key in solver.net.params.keys():
        if key in orig.net.params:
            for i in range(0,len(solver.net.params[key])):
                solver.net.params[key][i].data[...] = orig.net.params[key][i].data
    #print "end copy"
    return solver


def main():
    pycaffe_dir = os.path.dirname(__file__)

#    caffe.set_random_seed(10923)
    caffe.set_mode_gpu();
    if full_conv:
        w = 12
    else:
        w = 96
    batchsize = 4



    with open('teraFreeGAN.prototxt', 'w') as f:
        n = initOGAN(w, batchsize, 1)
        f.write(str(n))



    #start = 1000 hatte gute results
    #10000 ist auch gut
    #11500 sehr gut
    start = 0#60000

    solver = caffe.get_solver('solver_8.prototxt')
    resto = (start != 0)
    if resto:
        solver.restore('teraFreeGAN_adadelta_iter_'+str(start)+'.solverstate')
    #else:
    if False:
        orig = caffe.get_solver('solver_7.prototxt')
        orig.restore('tera2OGAN_phase2_adadelta_iter_'+str(2500)+'.solverstate')
        notparam =  [
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
"full3",
"addit1",
"addit2",
"addit3",
"addit4",
"addit5",
"last"
            ]
        for key in solver.net.params.keys():
            if key in orig.net.params:# and not (key in notparam):
                if solver.net.params[key][0].data.shape == orig.net.params[key][0].data.shape:
                    for i in range(0,len(solver.net.params[key])):
                        solver.net.params[key][i].data[...] = orig.net.params[key][i].data
                else:
                    solver.net.params[key][0].data[...] = orig.net.params[key][0].data[:, 0:1, :, :]
                    solver.net.params[key][1].data[...] = orig.net.params[key][1].data

    for i in range(0,8):
        solver.net.params["upsample"+"joinA"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    for i in range(0,16):
        solver.net.params["upsample"+"joinB"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    for i in range(0,24):
        solver.net.params["upsample"+"joinC"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    for i in range(0,32):
        solver.net.params["upsample"+"joinD"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])

    print solver.net.blobs["ganAconv11"].data.shape

    for i in range(0,64):
        print solver.net.params["last"][0].data[0,i]


    transformer = caffe.io.Transformer({'data': (1,1,w,w)})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 1.0)     


    with open('teraFreeGAN.prototxt', 'w') as f:
        n = initOGAN(w, batchsize, 2)
        f.write(str(n))

    solver2 = caffe.get_solver('solver_8.prototxt')

    copyParams(solver, solver2)

    for i in range(0,8):
        solver.net.params["upsample"+"joinA"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    for i in range(0,16):
        solver.net.params["upsample"+"joinB"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    for i in range(0,24):
        solver.net.params["upsample"+"joinC"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    for i in range(0,32):
        solver.net.params["upsample"+"joinD"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])

    with open('teraFreeGAN.prototxt', 'w') as f:
        n = initOGAN(w, batchsize, 1)
        f.write(str(n))

    solver = caffe.get_solver('solver_8.prototxt')

    copyParams(solver2, solver)

    for i in range(0,8):
        solver.net.params["upsample"+"joinA"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    for i in range(0,16):
        solver.net.params["upsample"+"joinB"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    for i in range(0,24):
        solver.net.params["upsample"+"joinC"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    for i in range(0,32):
        solver.net.params["upsample"+"joinD"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])


    img = caffe.io.load_image('newdata/test_data_3500_1000.bmp', color=False)
    width = img.shape[0]
    heigth = img.shape[1]
    img*=255.0
    img-=103.939
    print width
    print heigth

    if inmemory:
        for j in range(0, 16):
            offsetx = randint(0, width - (w + 1)) 
            offsety = randint(0, heigth - (w + 1)) 
            deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]
            solver.net.blobs['alldata'].data[j,:,:,:] = transformer.preprocess('data', deimg)
    p = 2
    last = start
    last = 0
    lastA = 0.7
    lastAB = 0.7
    if full_conv:
        w=68


    while True:
        
        
        if (start - last)>=400:
            last = start
            outp = solver.net.forward(blobs=["Aip3", "Alabels"], start="Adata")
            print outp["Alabels"]
            float_formatter = lambda x: "%.2f" % x
            np.set_printoptions(formatter={'float_kind':float_formatter})
            print 1.0 / (1.0 + np.exp(-outp["Aip3"][:,0]))
            np.set_printoptions()

            io.imshow(2.0*np.concatenate((solver.net.blobs["Atexture"].data[0,:,:,:].reshape((w,w)), solver.net.blobs["Atexture"].data[1,:,:,:].reshape(w,w), np.reshape(solver.net.blobs["Atexture"].data[2,:,:,:] ,(w,w))), axis=1))
            io.show()

            offsetx=25
            offsety=15
            deimg = img[(0+offsetx):(w+offsetx):1,(0+offsety):(w+offsety):1,0:1:1]
            dat = solver.net.blobs["Atexture"].data[0,:,:,:]
            io.imshow(2.0*np.concatenate((solver.net.blobs["Adata"].data[0,:,:,:].reshape((w,w)), np.reshape(dat ,(w,w))), axis=1))
            io.show()
            
        #autobalance training of generator and discriminator
        val= 0.6
        tval=0.7
        if lastA < 0.5:
            outp = solver.net.forward(blobs=["Aloss"])
            lastA = outp["Aloss"]
            print "check A " + str(lastA)
        if lastA >= 0.5:
            while tval > val:
                solver.step(1)
                start+=1
                tval = solver.net.blobs["Aloss"].data
                lastA = tval
                print "train A " + str(tval)
            copyParams(solver, solver2)
            
        val= 0.6
        tval=0.7
        iter = 0
        while tval > val:
            solver2.step(1)
            iter+=1
            start+=1
            if iter > 20:
                iter = 0
                copyParams(solver2, solver)
                outp = solver.net.forward(blobs=["Aloss"])
                if outp["Aloss"].data > val:
                    break
            tval = solver2.net.blobs["ABloss"].data
            lastAB = tval
            print "train AB " + str(tval)

        copyParams(solver2, solver)

        


if __name__ == "__main__":
    main()