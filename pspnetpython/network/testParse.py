
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt;
import sys
sys.path.insert(0, '/home/conteinerFiles/pspnet/PSPNet/python')
import caffe_a as caffe
import os
import argparse
import score
parser = argparse.ArgumentParser()

parser.add_argument("--path_solver", help="path to solver used")
parser.add_argument("--model_name", help="Model name used")
parser.add_argument("--snapshot_name", help="Snapshot folder name used")

args = parser.parse_args()

if args.path_solver:
    print "Solver path: " + args.path_solver

modelName = None
snapshotName = None


def run_pspnet_3C_modelPath(filename, caminhoIn, modelPath, protoPath, setLabel=True):
    caminhoFile = os.path.join(caminhoIn, filename + ".jpg")
    nameDivided = filename.split(".")

    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(caminhoFile)

    imSize = im.size
    im = im.resize((473,473))
    
    

    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))

    print caminhoFile
    
    net = caffe.Net(protoPath, modelPath, caffe.TEST)
    
    # shape for input (data blob is N x C x H x W), set data
    print "img entreda: "
    #print in_.shape
    print caminhoFile
    #descomentar para 3 canais
    #print "img depth: "
    #print  depthIn.shape
    net.blobs['color'].reshape(1, *in_.shape)
    net.blobs['color'].data[...] = in_

    label = cv2.imread('/home/conteinerFiles/skin-images/masks/{}.png'.format(filename),0)
    #label = label.resize((473,473))
    label = cv2.resize(label,(473,473))
    label = getBinaryImage(label)

    if(setLabel):
        sumLabel = 0
        label = label[np.newaxis, ...]

        net.blobs['label'].reshape(1, *label.shape)
        net.blobs['label'].data[...] = label

    return (net, label)

def getBinaryImage(labelImage):
	labelTemp = (labelImage != 0)
	labelImage = labelTemp * 1
	return labelImage


'''
def getBinaryImage(img, skinColor = 255):
        #img = cv2.resize(img,(200, 200))
        height, width = img.shape
        imgOut = np.zeros((height,width),dtype=np.uint8)
        for y in range(0,height): 
            for x in range(0,width): 
                if img[y,x] == skinColor:
                    imgOut[y,x] = 1

        return imgOut
'''


#val = np.loadtxt('/home/conteinerFiles/skin-images/val_one', dtype=str)
val = np.loadtxt('/home/conteinerFiles/skin-images/val', dtype=str)
caminhoIn = "/home/conteinerFiles/skin-images/skin-images"

if args.model_name:
    modelName = args.model_name
else:
    modelName = 'train_psp_100000_iterations_deeplabAug_correct'
if args.snapshot_name:
    snapshotName = args.snapshot_name
else:
    snapshotName = 'snapshot'

print "Model: " + modelName

modelPath = '/home/conteinerFiles/pspnet/pspnetpython/network/' + snapshotName + '/'+ modelName + '.caffemodel'
protoPath = '/home/conteinerFiles/pspnet/pspnetpython/network/test.prototxt'

logFile = open('/home/conteinerFiles/pspnet/pspnetpython/network/' + snapshotName + '/'+ modelName + '_acc.log', 'a+')

#val = np.array(['2940226'])
#val = np.array(['5147'])
print val
'''
for idx in val:
    net = run_pspnet_3C_modelPath(idx, caminhoIn, modelPath, protoPath)
    score.seg_tests(net, False, val, layer='conv6_interp_new')

'''

#net = run_pspnet_3C_modelPath('2940226', caminhoIn, modelPath, protoPath)
currentAcc = 0.0
for idx in val:
    net, label = run_pspnet_3C_modelPath(idx, caminhoIn, modelPath, protoPath, False)
    net.forward()

    out = net.blobs['conv6_interp_new'].data[0].argmax(axis=0).flatten()
    out = np.reshape(out, (out.shape[0], 1))
    label = label.flatten()
    label = np.reshape(label, (label.shape[0], 1))
    print out.shape
    print label.shape
    newAcc = float((np.dot(label.T,out) + np.dot(1-label.T,1-out))/float(out.size)*100)
    print idx + " acc: " + str(newAcc) + '%'
    logFile.write(idx + " acc: " + str(newAcc) + '%' + '\n')
    currentAcc = currentAcc + newAcc

totalAcc = float(currentAcc/float(val.size))
#print ('Accuracy: %d' %  + '%')
print "Total acc: " + str(totalAcc) + '%'
logFile.write("Total acc: " + str(totalAcc) + '%' + '\n')
#print ('Accuracy: %d' % float((np.dot(label.T,out) + np.dot(1-label.T,1-out))/float(out.size)*100) + '%')
#print net.blobs['label'].data[0, 0].shape
'''
print "label shape"
labelOut =  net.blobs['label'].data[0, 0]
for y in range(0, labelOut.shape[0]):
    for x in range(0, labelOut.shape[1]):
        if labelOut[y, x] > 0:
            print labelOut[y, x]
print "sum label: " + str(sumLabel)
'''
#score.do_seg_tests(net, 1, False, val, layer='conv6_interp_new')



#score.seg_tests(net, False, val, layer='conv6_interp_new')