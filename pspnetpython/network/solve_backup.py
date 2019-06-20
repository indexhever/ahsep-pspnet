#import caffe
import sys
#sys.path.insert(0, '/home/hmelo/caffe/python')
# uncomment this on Speed
#sys.path.insert(0, '/home/hmelo/caffe2/python')
#sys.path.insert(0, '/home/hmelo/MNC/caffe-mnc/python')
#import caffe_a as caffe
#import caffe_b as caffe
#import caffe_c as caffe
#import caffe_d as caffe
#import caffe_e as caffe
import caffe
#import caffe_f as caffe

import surgery, score

import numpy as np
import os
#import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../ilsvrc-nets/vgg16-fcn.caffemodel'
#uncommment for Speed

#base_net = caffe.Net('/home/hmelo/fcnd/ilsvrc-nets/vgg16fcn.prototxt', '/home/hmelo/fcnd/ilsvrc-nets/vgg16fcn.prototxt.caffemodel',
 #       caffe.TEST)
base_net = caffe.Net('/home/conteinerFiles/pspnet/pspnetpython/ilsvrc-nets/vgg16fcn.prototxt', '/home/conteinerFiles/pspnet/pspnetpython/ilsvrc-nets/vgg16fcn.prototxt.caffemodel',
       caffe.TEST)

# init
#caffe.set_device(int(sys.argv[1]))
caffe.set_device(0)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

#solver = caffe.SGDSolver('/home/hmelo/fcnd/nyud-fcn32s-color-d/solver.prototxt')
solver = caffe.SGDSolver('/home/conteinerFiles/pspnet/pspnetpython/network/solver.prototxt')
surgery.transplant(solver.net, base_net)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)


#descomentar para 4 canais
'''
solver.net.params['conv1_1_bgrd'][0].data[:, :3] = base_net.params['conv1_1'][0].data
solver.net.params['conv1_1_bgrd'][0].data[:, 3] = np.mean(base_net.params['conv1_1'][0].data, axis=1)
solver.net.params['conv1_1_bgrd'][1].data[...] = base_net.params['conv1_1'][1].data
'''
# 3 canais

solver.net.params['conv1_1_bgrd'][0].data[:, :3] = base_net.params['conv1_1'][0].data
#solver.net.params['conv1_1_bgrd'][0].data[:, 3] = np.mean(base_net.params['conv1_1'][0].data, axis=1)
solver.net.params['conv1_1_bgrd'][1].data[...] = base_net.params['conv1_1'][1].data


del base_net

# scoring
#test = np.loadtxt('../data/nyud/test.txt', dtype=str)
#uncomment for Speed
#test = np.loadtxt('/home/hmelo/images/skin-images/test', dtype=str)
test = np.loadtxt('/home/conteinerFiles/skin-images/test', dtype=str)
#uncomment for Speed
#val = np.loadtxt('/home/hmelo/images/skin-images/val', dtype=str)
val = np.loadtxt('/home/conteinerFiles/skin-images/val', dtype=str)

for _ in range(50):
    solver.step(2000)
    score.seg_tests(solver, False, val, layer='score')
