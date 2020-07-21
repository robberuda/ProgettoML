import cv2 as cv
import numpy as np
from palette import colors
import time
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__)) # init path (pyCharm)

cap = cv.VideoCapture(0) # webcam n#0
whT = 320                # use a sqare image
confThreshold = 0.5      # our value of threshold
nmsThreshold = 0.2       # non maximum suppression
 

# Load Coco Names of classes
classesFile = os.path.join(THIS_FOLDER,"coco.names.txt" )
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

print(classNames)

# Model Files downloaded from YOLO website https://pjreddie.com/darknet/yolo/:
modelConfiguration = os.path.join(THIS_FOLDER,"yolov3-320.cfg")
modelWeights = os.path.join(THIS_FOLDER,"yolov3-320.weights")

''' create the CNN with ".cfg" configuration file and ".weights" weights of 
    train '''
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) #scelta backend (software)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)      #scelta target  (hardware)


''' this function take in input the output of CNN and create boxes over image '''
def findObjects(outputs,img):
    hT, wT, cT = img.shape # image's dimension
    bbox = []              # list of boxes
    classIds = []          # list of ids
    confs = []             # list of confidences
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                ''' use de output vector of yolo for add a box in 'bbox' list: id of 
                object, dimension of box, cordinates in the image and the confidance '''
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
                
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]  #i = indicate how much object in the image
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        color = colors[classIds[i]] #load different colors for boxes

        '''add  box and lable'''
        cv.rectangle(img, (x, y), (x+w,y+h), color, 2)
        labelLenght = len(f'{classNames[classIds[i]].upper()}')
        cv.rectangle(img, (x-1, y), (x+11*labelLenght+62,y-25), color, -2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x+2, y-7), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
if __name__=="__main__": 
    start_time = time.time()
    nFps = 6
    frame = 0
    fps = 'initialization FPS'

    while True:
        success, img = cap.read()#capture image by webcam
        #convert image in blob (binary large object)
        blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames() #take name of all network's layers
        # getUnconnectedOutLayers take index of out layers of CNN, in particular are three: 200, 227 and 254(last layer)
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames) #outputsNames=(yolo_82,yolo94,yolo106)
        findObjects(outputs,img)
        frame += 1
        TIME = time.time() - start_time

        if frame > nFps:
            fps='FPS: '+str("{0:.2f}".format(frame/TIME))
            # print("FPS:", frame/TIME)
            frame = 0
            start_time = time.time()

        cv.putText(img, fps, (2, 18), cv.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
        cv.imshow('Image', img) #show new image with
        cv.waitKey(1)