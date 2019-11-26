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
from randomrot import * 
from resizelayer import *
import cv2

import caffe
from caffe import layers as L
from caffe import params as P

inmemory = False
upsample = True
global_stat = True
full_conv = True

def convBlock(name, opn, n, input, train=True):
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
    convolution_param = dict(num_output=opn, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    if full_conv:
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

def joinBlock(name, opn, n, inputA, inputB, train=True):
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

def LowAGAN(w, batchsize, n):
    #input w = 11 damit output 10 <= 8*10 ist maximum
    #input w = 16 damit output 20 <= 4*20 ist maximum

    level = 2
    listofsizes =  []
    if full_conv:
        listofsizes = [w]
    
        for i in range(0, level-1):
            alast = listofsizes[i]
            listofsizes.append((alast - 4)*2)
        listofsizes[0] -= 4


    transform_param = dict(mirror=False, crop_size=w, scale=1., mean_value=103.939)
    if full_conv:
        transform_param = dict(mirror=False, crop_size=120, scale=1., mean_value=103.939)
    n.Adata, n.Anothing = L.ImageData(transform_param=transform_param, source='datasource.txt', 
                            is_color=False, shuffle=True, batch_size=batchsize, ntop=2)
    n.Aresize = L.Python(n.Adata, python_param=dict(module='resizelayer', layer='ResizeData'), param_str=str(4))
    n.Acropped = L.Python(n.Aresize, python_param=dict(module='randomrot', layer='RandomRotLayer'), param_str=str(listofsizes[level -1] - 4))

    codings = [8, 16, 24, 32, 40]

    d=w
    outname = ""
    for i in range(0,level):
        if full_conv:
            n["AZrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, listofsizes[level-1 -i] + 4, listofsizes[level-1 -i] + 4])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)
        else:
            n["AZrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, d, d])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)
        n, outname = convBlock("AconvA"+str(i), codings[0], n, "AZrand_"+str(i), train=False)
        d /= 2

    n, outname = joinBlock("AjoinA", codings[0], n, outname, 'gelu'+'AconvA0'+'_3', train=False)

    n, outname = convBlock("AconvB", codings[1], n, outname, train=False)
    convolution_param = dict(num_output=1, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n["Atexture"] = L.Convolution(n[outname], convolution_param=convolution_param, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], name="Atextureparam")

    #0 => blurdat
    #1 => data
    n.Aswap, n.Alabels = L.Python(n["Atexture"], n.Acropped, python_param=dict(module='swaplayer', layer='SwapLayer'), propagate_down=[False, False], ntop=2)

    if full_conv:
        n.Anoise = L.DummyData(shape=[dict(dim=[batchsize, 1, listofsizes[level -1] - 4, listofsizes[level -1] - 4])], data_filler=dict(type='gaussian', std=2.0), ntop=1)
    else:
        n.Anoise = L.DummyData(shape=[dict(dim=[batchsize, 1, w, w])], data_filler=dict(type='gaussian', std=2.0), ntop=1)

    n.Ainp = L.Eltwise(n.Aswap, n.Anoise, eltwise_param={'operation':1})

    #GAN network
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv1 = L.Convolution(n.Ainp, convolution_param=convolution_param)
    n.ganAconv1 = L.ReLU(n.ganAconv1, negative_slope=0.1)
    
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv2 = L.Convolution(n.ganAconv1, convolution_param=convolution_param)
    n.ganAconv2 = L.ReLU(n.ganAconv2, negative_slope=0.1)
    
    convolution_param = dict(num_output=32, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganAconv3 = L.Convolution(n.ganAconv2, convolution_param=convolution_param)
    n.ganAconv3 = L.ReLU(n.ganAconv3, negative_slope=0.1)

    convolution_param = dict(num_output=32, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv4 = L.Convolution(n.ganAconv3, convolution_param=convolution_param)
    n.ganAconv4 = L.ReLU(n.ganAconv4, negative_slope=0.1)
    
    convolution_param = dict(num_output=32, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganAconv5 = L.Convolution(n.ganAconv4, convolution_param=convolution_param)
    #n.ganAconv5 = L.BatchNorm(n.ganAconv5, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv5 = L.ReLU(n.ganAconv5, negative_slope=0.1)
    convolution_param = dict(num_output=32, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganAconv6 = L.Convolution(n.ganAconv5, convolution_param=convolution_param)
    #n.ganAconv6 = L.BatchNorm(n.ganAconv6, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv6 = L.ReLU(n.ganAconv6, negative_slope=0.1)

    convolution_param = dict(num_output=32, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganAconv7 = L.Convolution(n.ganAconv6, convolution_param=convolution_param)
    #n.ganAconv7 = L.BatchNorm(n.ganAconv7, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganAconv7 = L.ReLU(n.ganAconv7, negative_slope=0.1)
    n.ganAconv7_pool = L.Pooling(n.ganAconv7, global_pooling=True, pool=P.Pooling.AVE)

    n.Aip3 = L.InnerProduct(n.ganAconv7_pool, num_output=1, weight_filler=dict(type='xavier'), name="last")
    
    n.Aloss = L.SigmoidCrossEntropyLoss(n.Aip3, n.Alabels)

    return n

def LowABGAN(w, batchsize, n):
    n.ABnothing = L.DummyData(shape=[dict(dim=[batchsize, 1, 1, 1])], data_filler=dict(type='constant'), ntop=1)
    n.ABlabels = L.Python(n.ABnothing, python_param=dict(module='destroy', layer='DestroyLayer'))
    codings = [8, 16, 24, 32, 40]

    level = 2
    listofsizes =  []
    if full_conv:
        listofsizes = [w]
    
        for i in range(0, level-1):
            alast = listofsizes[i]
            listofsizes.append((alast - 4)*2)
        listofsizes[0] -= 4

    d=w
    outname = ""
    for i in range(0,level):
        if full_conv:
            n["ABZrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, listofsizes[level-1 -i] + 4, listofsizes[level-1 -i] + 4])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)
        else:
            n["ABZrand_"+str(i)] = L.DummyData(shape=[dict(dim=[batchsize, 1, d, d])], data_filler=dict(type='uniform',min=0., max=255.), ntop=1)
        n, outname = convBlock("ABconvA"+str(i), codings[0], n, "ABZrand_"+str(i), train=True)
        d /= 2

    n, outname = joinBlock("ABjoinA", codings[0], n, outname, 'gelu'+'ABconvA0'+'_3', train=True)

    n, outname = convBlock("ABconvB", codings[1], n, outname, train=True)
    convolution_param = dict(num_output=1, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n["ABtexture"] = L.Convolution(n[outname], convolution_param=convolution_param, name="Atextureparam")

    #GAN network
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv1 = L.Convolution(n["ABtexture"], param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
    n.ganABconv1 = L.ReLU(n.ganABconv1, negative_slope=0.1)
    
    convolution_param = dict(num_output=16, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv2 = L.Convolution(n.ganABconv1, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
    n.ganABconv2 = L.ReLU(n.ganABconv2, negative_slope=0.1)
    
    convolution_param = dict(num_output=32, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganABconv3 = L.Convolution(n.ganABconv2, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
    n.ganABconv3 = L.ReLU(n.ganABconv3, negative_slope=0.1)

    convolution_param = dict(num_output=32, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv4 = L.Convolution(n.ganABconv3, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
    n.ganABconv4 = L.ReLU(n.ganABconv4, negative_slope=0.1)
    
    convolution_param = dict(num_output=32, kernel_size=3, stride=1, pad=1, weight_filler = dict(type='xavier'))
    n.ganABconv5 = L.Convolution(n.ganABconv4, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
    #n.ganAconv5 = L.BatchNorm(n.ganAconv5, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv5 = L.ReLU(n.ganABconv5, negative_slope=0.1)
    convolution_param = dict(num_output=32, kernel_size=2, stride=2, pad=0, weight_filler = dict(type='xavier'))
    n.ganABconv6 = L.Convolution(n.ganABconv5, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
    #n.ganAconv6 = L.BatchNorm(n.ganAconv6, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv6 = L.ReLU(n.ganABconv6, negative_slope=0.1)

    convolution_param = dict(num_output=32, kernel_size=1, stride=1, pad=0, weight_filler = dict(type='xavier'))
    n.ganABconv7 = L.Convolution(n.ganABconv6, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], convolution_param=convolution_param)
    #n.ganAconv7 = L.BatchNorm(n.ganAconv7, use_global_stats=global_stat, name=allparamnames.pop(0))#, param=[{"lr_mult":0},{"lr_mult":0},{"lr_mult":0}])
    n.ganABconv7 = L.ReLU(n.ganABconv7, negative_slope=0.1)
    n.ganABconv7_pool = L.Pooling(n.ganABconv7, global_pooling=True, pool=P.Pooling.AVE)

    n.ABip3 = L.InnerProduct(n.ganABconv7_pool, param=[dict(lr_mult=0, decay_mult= 0),dict(lr_mult=0, decay_mult= 0)], num_output=1, weight_filler=dict(type='xavier'), name="last")
    
    n.ABloss = L.SigmoidCrossEntropyLoss(n.ABip3, n.ABlabels)

    return n

def initFreeLowGAN(w, batchsize, p=1):
    n = caffe.NetSpec()
    if p == 1:
        n = LowAGAN(w, batchsize, n)
    else:
        n = LowABGAN(w, batchsize, n)
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
        w = 16
    else:
        w = 20
    batchsize = 32



    with open('teraFreeLowGAN.prototxt', 'w') as f:
        n = initFreeLowGAN(w, batchsize, 1)
        f.write(str(n))


    #start = 1000 hatte gute results
    #10000 ist auch gut
    #11500 sehr gut
    start = 16000#60000

    solver = caffe.get_solver('solver_9.prototxt')
    resto = (start != 0)
    if resto:
        solver.restore('teraFreeLowGAN_adadelta_iter_'+str(start)+'.solverstate')

    for i in range(0,8):
        solver.net.params["upsampleparam"+"joinA"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    

    for i in range(0,32):
        print solver.net.params["last"][0].data[0,i]


    transformer = caffe.io.Transformer({'data': (1,1,w,w)})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 1.0)     


    with open('teraFreeLowGAN.prototxt', 'w') as f:
        n = initFreeLowGAN(w, batchsize, 2)
        f.write(str(n))

    solver2 = caffe.get_solver('solver_9.prototxt')

    copyParams(solver, solver2)

    for i in range(0,8):
        solver.net.params["upsampleparam"+"joinA"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    
    with open('teraFreeLowGAN.prototxt', 'w') as f:
        n = initFreeLowGAN(w, batchsize, 1)
        f.write(str(n))

    solver = caffe.get_solver('solver_9.prototxt')

    copyParams(solver2, solver)

    for i in range(0,8):
        solver.net.params["upsampleparam"+"joinA"][0].data[i,i,:,:] = np.array([[1.,1.],[1.,1.]])
    

    p = 2
    last = start
    last = 0
    lastA = 0.7
    lastAB = 0.7
    if full_conv:
        w=20


    while True:
        
        
        if (start - last)>=400 and False:
            last = start
            outp = solver.net.forward(blobs=["Aip3", "Alabels"], start="Adata")
            print outp["Alabels"]
            float_formatter = lambda x: "%.2f" % x
            np.set_printoptions(formatter={'float_kind':float_formatter})
            print 1.0 / (1.0 + np.exp(-outp["Aip3"][:,0]))
            np.set_printoptions()

            io.imshow(2.0*np.concatenate((solver.net.blobs["Atexture"].data[0,:,:,:].reshape((w,w)), solver.net.blobs["Atexture"].data[1,:,:,:].reshape(w,w), np.reshape(solver.net.blobs["Atexture"].data[2,:,:,:] ,(w,w))), axis=1))
            io.show()

            dat = solver.net.blobs["Atexture"].data[0,:,:,:]
            io.imshow(2.0*np.concatenate((solver.net.blobs["Acropped"].data[0,:,:,:].reshape((w,w)), np.reshape(dat ,(w,w))), axis=1))
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