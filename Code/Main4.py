import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import Model
import filterpy
from filterpy.kalman import KalmanFilter
import Track
import Traker
import Detection
import kalman_filter


# Initialize YOLOv4 object detector
net = cv2.dnn.readNetFromDarknet('F:/CV/Last2/yolov4_Config/yolov4.cfg','F:/CV/Last2/yolov4_weights/yolov4.weights')
classes = []
with open('classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]


layer_names = net.getLayerNames()
tracker = Traker.Traker(max_age=60,max_dist=0.7)

# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize ResNet50 feature extractor
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = resnet
# model = Model(inputs=resnet.input, outputs=resnet.get_layer('avg_pool').output)

# Initialize Kalman Filters for object tracking
num_objects = 1  # Set the number of objects to track


# Open video capture device
cap = cv2.VideoCapture('F:/CV/Last2/input/obj2.mp4')
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

output_video = cv2.VideoWriter('F:/CV/Last2/output/Maulik_Occ2.mp4v', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, frame_size)

while True:
    # Read frame from video capture device
    ret, frame = cap.read()
    
    if not ret:
        print("Video Finished")
        break
    
    if ret:
        # Perform object detection using YOLOv4
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())
        class_ids = []
        confidences = []
        boxes = []
        Dets = []
        # tot_detection = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    prev_h = h
                    prev_w = w
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    # self,xywh,score,feature,class_name,mean,covariance,kf
               
            indices = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
            
            
            # Extract features using ResNet50 for each object
            
            features_list = []
            tot_detections = []
            tot_scores = []
            tot_class_names = []
            
            if(len(indices)>0):
                for i in indices.flatten():
                    box = boxes[i]
                    center_x,center_y,width,height = int(box[0] + box[2]/2), int(box[1] + box[3]/2), box[2],box[3]
                    roi = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                    if roi.shape[0]!=0 and roi.shape[1]!=0:
                        roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
                        roi = np.expand_dims(roi, axis=0)
                        features = model.predict(preprocess_input(roi))
                        # print(features)
                        features_list.append(features)
                        tot_detections.append([center_x,center_y,width,height])
                        tot_scores.append(confidences[i])
                        tot_class_names.append((classes[class_ids[i]]))
                        
                        
                
                # def __init__(self,xywh,score,feature,class_name,mean,covariance,kf)
            Dets = []
            for feat, dete, sc,cN in zip(features_list,tot_detections,tot_scores,tot_class_names):
                # print(type(feat))
                Dets.append(Detection.Detection(dete,sc,feat,"Hello",1,2,kalman_filter.KalmanF()))
            
            # if len(Dets)!=0:
            #     print(Dets[0].)
            total_Detection_To_show = tracker.update(Dets)
            # print("WHYYYYY!!!")
            print("-------------")
            print(total_Detection_To_show)
            print("-------------")
                
                
        
    
    # Draw bounding boxes and labels for each object
        for box in total_Detection_To_show:
            if box is not None:
                x,y,w,h = box
                # label = str(classes[class_ids[i]])
             # confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(x-w), int(y-h)), (int(x + w), int(y + h)), color, 2)
                # cv2.putText(frame," " + str(round(confidence, 2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
       

    
    
    # Display the resulting frame
    
    cv2.imshow('frame', frame)
    output_video.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
