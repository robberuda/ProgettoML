import cv2 as cv
import numpy as np
from palette import colors
import time
 
cap = cv.VideoCapture(0) #webcam n#0
whT = 320 #risoluzione quadrata
confThreshold = 0.4 # soglia riconoscimento
nmsThreshold = 0.2  #non maximum suppression,fattore che elimina i box di troppo
 
#### LOAD MODEL
## Coco Names
classesFile = "coco.names.txt"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
## Model Files
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"


"""creazione della rete convoluzionale con il file .cfg di configurazione
   e il file dei pesi di YOLO preso dal sito """
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV) #scelta backend (software)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)      #scelta target  (hardware)


""" questa funzione prende in input un immagine e restituisce l'immagine con i
    riquadri che classificano gli oggetti, con l'etichetta e la precisione """
def findObjects(outputs,img):
    hT, wT, cT = img.shape #dimensioni immagine in input
    
    bbox = []    #list of boxes
    classIds = []#list of ids
    confs = []   #list of confidences
    
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                """aggiunge alla lista 'bbox' i box che saranno centrati in x 
                e y, e avranno dimensione w e h.
                Inoltre aggiunge alle liste 'classIds' e 'confs' gli id degli 
                oggetti nell'immagine e le relative confidenze """
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
                
    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    
    """ """
    for i in indices:
        i = i[0] # i = quanti oggetti nella figura
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        color = colors[classIds[i]] #cambia colore per ogni oggetto in figura
        
        labelLenght = len(f'{classNames[classIds[i]].upper()}')
        cv.rectangle(img, (x-1, y), (x+11*labelLenght+62,y-25), color, -2) #rettangolo nome
        
        cv.rectangle(img, (x, y), (x+w,y+h), color, 2)
        cv.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                  (x+2, y-7), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
 
    
if __name__=="__main__": 
    start_time = time.time()
    nFps = 6
    frame= 0
    fps='barbabietola'
    
    while True:
        success, img = cap.read()
     
        blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT),
                                    [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)
        # print (net)
        findObjects(outputs,img)
        

        frame += 1
        TIME = time.time() - start_time
        if frame > nFps:
            fps='FPS: '+str("{0:.2f}".format(frame/TIME))
            # print("FPS:", frame/TIME)
            frame = 0
            start_time = time.time()
        
        cv.putText(img, fps, (2, 18), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv.imshow('Image', img) #show new image with 
        cv.waitKey(1)