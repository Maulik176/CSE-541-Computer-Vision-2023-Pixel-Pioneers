import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import Model
import filterpy
from filterpy.kalman import KalmanFilter

# Initialize YOLOv4 object detector
net = cv2.dnn.readNetFromDarknet('F:/CV/Last2/yolov4_Config/yolov4.cfg','F:/CV/Last2/yolov4_weights/yolov4.weights')
classes = []
with open('classes.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Initialize ResNet50 feature extractor
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = resnet
# model = Model(inputs=resnet.input, outputs=resnet.get_layer('avg_pool').output)

# Initialize Kalman Filters for object tracking
num_objects = 1  # Set the number of objects to track

kf_list = []

for i in range(num_objects):
    kf = KalmanFilter(dim_x=4,dim_z=2)
    kf.x = np.zeros(4)  # Initial state (x, y, vx, vy)
    kf.P = np.eye(4) * 1000  # Initial state covariance
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # Measurement matrix
    kf.R = np.eye(2) * 10  # Measurement noise covariance
    kf.Q = np.eye(4) * 0.1  # Process noise covariance
    kf_list.append(kf)


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
                    prev_conf = confidence
                    prev_class_id = class_id
                    x = center_x - w // 2
                    y = center_y - h // 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        
        
        
        # Perform NMS
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        features_list = []
        if(len(indices) > 0):
            for i in indices.flatten():
                box = boxes[i]
                prev_class_id = class_ids[i]
                prev_conf = confidences[i]
                roi = frame[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
                if roi.shape[0]!=0 and roi.shape[1]!=0:
                    roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
                    roi = np.expand_dims(roi, axis=0)
                    features = model.predict(preprocess_input(roi))
                    features_list.append(features)
        
        # Predict next position of each object using Kalman Filter
        bbox_pred_list = []
        
        for i, kf in enumerate(kf_list):
            kf.predict()
            if i >= len(features_list):
                bbox_pred_list.append(None)
            else:
                x, y, w, h = boxes[i]
                prev_h = h
                prev_w = w
                
            if(len(boxes)>=4 and len(boxes[0])>=4):
                # Get current object center coordinates
                print("Box len",len(boxes))
                print("Box len0",len(boxes[0]))
                x_center = boxes[i][0] + boxes[i][2] / 2
                y_center = boxes[i][1] + boxes[i][3] / 2
                prev_h = h
                prev_w = w
                # Update Kalman Filter with current object center coordinates
                z = np.array([x_center, y_center])
                kf.update(z)

                # Predict next object center coordinates using Kalman Filter
            x_pred = kf.x[0]
            y_pred = kf.x[1]
            if (len(boxes)>=4 and len(boxes[0])>=4):
                bbox_width = boxes[i][2]
                bbox_height = boxes[i][3]
                prev_h = bbox_height
                prev_w = bbox_width
            else:
                bbox_width = prev_w
                bbox_height = prev_h

            bbox_pred = [int(x_pred - bbox_width/2), int(y_pred - bbox_height/2), bbox_width, bbox_height]
            bbox_pred_list.append(bbox_pred)
    
    # Draw bounding boxes and labels for each object
        for box in bbox_pred_list:
            if box is not None:
                x,y,w,h = box
                # label = str(classes[class_ids[i]])
             # confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame,classes[prev_class_id]+" "+str(round(prev_conf, 2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
       

    
    
    # Display the resulting frame
    
    cv2.imshow('frame', frame)
    output_video.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
