
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
import time
parser = argparse.ArgumentParser()

parser.add_argument("--path_solver", help="path to solver used")
parser.add_argument("--model_name", help="Model name used")
parser.add_argument("--snapshot_name", help="Snapshot folder name used")
parser.add_argument("--base_name", help="Base file name used (train, val or test)")
parser.add_argument("--run_acc", help="Run accuracy (True or nothing)")
parser.add_argument("--is_debug", help="Run in debug (True or nothing)")
parser.add_argument("--path_in", help="Path to run test")

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

    label = cv2.imread('/home/conteinerFiles/skin-images/masks/{}.pbm'.format(filename),0)
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

def LoadNet(protoPath, modelPath):
    net = caffe.Net(protoPath, modelPath, caffe.TEST)
    return net

def PreProcessInputImage(inputImagePath):
    im = Image.open(inputImagePath)
    imSize = im.size
    im = im.resize((473,473))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((104.00698793,116.66876762,122.67891434))
    in_ = in_.transpose((2,0,1))
    return in_

def RunNet(net, inputImage):
    net.blobs['color'].reshape(1, *inputImage.shape)
    net.blobs['color'].data[...] = inputImage
    net.forward()
    out = net.blobs['conv6_interp_new'].data[0].argmax(axis=0)
    return out

def ConfMatrix(resultImg, maskImg, totalArm, totalNotArm, predArmAndIsArm, predArmAndIsNotArm, predNotArmIsArm, predNotArmIsNotArm, totalPixels, backColorSegMap=0, frontColorLabel=1):
    height, width = maskImg.shape
    for y in range(0,height): 
        for x in range(0,width): 
            #print resultImg[y,x]
            #backgroundColor = 14
            #backgroundColor = 30
            if maskImg[y,x] == frontColorLabel:
                totalArm = totalArm + 1
            else:
                totalNotArm = totalNotArm + 1
            #print "color: "
            #print resultImg[y,x]
            # when resultImage value is arm and maskImage value is arm
            if resultImg[y,x] != backColorSegMap and maskImg[y,x] == frontColorLabel:
                predArmAndIsArm = predArmAndIsArm + 1
            
            # when resultImage value is arm and maskImage value is not arm
            if resultImg[y,x] != backColorSegMap and maskImg[y,x] != frontColorLabel:
                predArmAndIsNotArm = predArmAndIsNotArm + 1

            # when resultImage value is not arm and maskImage value is arm
            if resultImg[y,x] == backColorSegMap and maskImg[y,x] == frontColorLabel:
                predNotArmIsArm = predNotArmIsArm + 1

            # when resultImage value is not arm and maskImage value is not arm
            if resultImg[y,x] == backColorSegMap and maskImg[y,x] != frontColorLabel:
                predNotArmIsNotArm = predNotArmIsNotArm + 1

            totalPixels = totalPixels + 1
    return (totalArm, totalNotArm, predArmAndIsArm, predArmAndIsNotArm, predNotArmIsArm, predNotArmIsNotArm, totalPixels)

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

#caminhoIn = "/home/conteinerFiles/skin-images/skin-images"
caminhoIn = "/home/conteinerFiles/fcnhs/codes/frames"


if args.model_name:
    modelName = args.model_name
else:
    modelName = 'train_psp_100000_iterations_deeplabAug_correct'
if args.snapshot_name:
    snapshotName = args.snapshot_name
else:
    snapshotName = 'snapshot'
if args.base_name:
    baseName = args.base_name
else:
    baseName = "test"
if args.run_acc:
    runAcc = True
else:
    runAcc = False
if args.is_debug:
    isDebug = True
else:
    isDebug = False
if args.path_in:
    path_in = args.path_in
else:
    isDebug = ""

print "Model: " + modelName
outPath = "/home/conteinerFiles/FinalResults/Tests/" + snapshotName + "_GTEA_speedTest"
modelPath = '/home/conteinerFiles/pspnet/pspnetpython/network/' + snapshotName + '/'+ modelName + '.caffemodel'
#protoPath = '/home/conteinerFiles/pspnet/pspnetpython/network/test.prototxt'
protoPath = '/home/conteinerFiles/pspnet/pspnetpython/network/deploy.prototxt'
confMatrixLogFile = open('/home/conteinerFiles/pspnet/pspnetpython/network/' + snapshotName + '/'+ modelName + '_' + baseName + '_GTEA_timeTest.log', 'a+')
#net = run_pspnet_3C_modelPath('2940226', caminhoIn, modelPath, protoPath)

net = LoadNet(protoPath, modelPath)
caminhoIn = path_in
#myFiles = os.listdir(caminhoIn)
myFiles = np.loadtxt(caminhoIn, dtype=str)
totalTime = 0.0
count = 0
for idx in myFiles:
    count = count + 1
    inputImagePath = "/home/conteinerFiles/skin-images/skin-images/" + idx + ".jpg"
    print inputImagePath
    inputImage = PreProcessInputImage(inputImagePath)
    startTimeRun = time.time()
    out = RunNet(net, inputImage)
    endTimeRun = time.time()
    currentTime = (endTimeRun - startTimeRun)
    totalTime = totalTime + currentTime
    print "Time run: " + str(currentTime)
    confMatrixLogFile.write("time " + idx + " = " + str(currentTime) + '\n')
    plt.imsave(os.path.join(outPath, idx + "_473x473_out.jpg"), out)
    im = Image.open(inputImagePath)
    im.save(os.path.join(outPath, idx+".jpg"))

total = totalTime / float(count)

confMatrixLogFile.write("Average time " + " = " + str(total) + '\n')
confMatrixLogFile.write("Total time " + " = " + str(totalTime) + '\n')