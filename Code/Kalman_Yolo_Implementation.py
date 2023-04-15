import cv2
import numpy as np
from collections import deque
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean



# Load YOLOv4 weights and configuration
net = cv2.dnn.readNetFromDarknet("F:/CV/Last2/yolov4_Config/yolov4.cfg", "F:/CV/Last2/yolov4_weights/yolov4.weights")


# Load ResNet50 model
resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = resnet

# Define colors for different objects
colors = np.random.uniform(0, 255, size=(100, 3))

# Define parameters for DeepSORT
max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 1.0
Max_age = 10

# Define class for tracking objects
def defineKalmanforme():
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
    kf.Q = np.eye(4) * 0.1
    
    return kf




class Track:
    def __init__(self, bbox, features,kf):
        self.bbox = bbox
        self.features = features
        self.track_id = None
        self.hits = 1
        self.age = 1
        self.kf = kf
        self.track_state = True

    def predict(self):
        self.age += 1
        self.hits = 0
        self.kf.predict()
        self.bbox[0] = self.kf.x[0]
        self.bbox[1] = self.kf.x[1]

    def update(self, bbox, features):
        self.bbox = bbox
        self.features = features
        self.age = 0
        self.hits += 1
        z = np.array([bbox[0],bbox[1]])
        self.kf.update(z)
        
    def missing(self):
        if self.age>Max_age:
            self.track_state = False

# Define function to compute IoU (Intersection over Union) between two bounding boxes
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou



def solve_Hungarian(cost_matrix):
    row_indices,column_indices = linear_sum_assignment(cost_matrix)
    return row_indices,column_indices


def distance(A1,A2):
    return euclidean(A1, A2)

def Match(tracks,detections):
    cost_matrix = np.zeros((len(detections),len(tracks)))
    for i in range(len(detections)):
        for j in range(len(tracks)):
            A1 = detections[i][1]
            A2 = tracks[j].features
            cost_matrix[i][j] = distance(A1,A2)
    print("Cost: ",cost_matrix)
            
    matched_tracks = []
    matched_dets = []
    unmatched_tracks = []
    unmatched_dets = []
    row_indices,column_indices = solve_Hungarian(cost_matrix)
    for r,c in zip(row_indices,column_indices):
        if cost_matrix[r][c] > 0.5:
            matched_tracks.append(c)
            matched_dets.append(r)
        else:
            unmatched_dets.append(r)
            
            
    return matched_tracks,matched_dets,unmatched_dets


    


cap = cv2.VideoCapture("F:\CV\Last2\input\obj2.mp4")
if not cap.isOpened():
    print("Could not open video file.")

# Define variables for tracking objects
track_id_count = 0
tracks = []
memory = deque(maxlen=100)

    # Process video frames
while True:
    # Read frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Get image dimensions
    img_height, img_width, _ = frame.shape

    #  Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set input to the network
    net.setInput(blob)

    # Forward pass through the network
    # output = net.forward(output_layer_names)
    output = net.forward(net.getUnconnectedOutLayersNames())


    # Extract bounding boxes, confidence scores, and class IDs
    boxes = []
    confidences = []
    class_ids = []
    for i in range(len(output)):
        for detection in output[i]:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                width = int(detection[2] * img_width)
                height = int(detection[3] * img_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to remove overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Extract features using ResNet50
    detections = []
    if (len(indices)>0):
        for i in indices.flatten():
            box = boxes[i]
            x_center, y_center, width, height = box[0] + width / 2, box[1] + height / 2, width, height
            roi = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            img = image.img_to_array(roi)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            feature_vector = model.predict(img)
            detections.append((box, feature_vector))

    
    # Predicting every track
    for track in tracks:
        track.predict()
    
    # Update tracks with the help of APP vec
    matched_tracks_idx,matched_dets_idx ,unmatched_dets_idx = Match(tracks,detections)
    Matched_tracks2 = []
    Unmatched_track2 = []
    New_detections2 = []
    
    for r,c in zip(matched_tracks_idx,matched_dets_idx):
        Matched_tracks2.append(tracks[r])
        tracks[r].update(detections[c][0],detections[c][1])
    
    k = 0
    for track in tracks:
        if k not in matched_tracks_idx:
            Unmatched_track2.append(track)
        k+=1
        
    
    
    for i in unmatched_dets_idx:
        kf = defineKalmanforme()
        bbox = detections[i][0]
        features = detections[i][1]
        z = np.array([detections[i][0][0],detections[i][0][1]])
        kf = defineKalmanforme()
        kf.update(z)
        track = Track(bbox, features,kf)
        track.track_id = track_id_count
        track_id_count += 1
        New_detections2.append(track)

    # Initial Condition.  
    if len(tracks) == 0 and len(detections) != 0:
        for d in detections:
            bbox = d[0]
            features = d[1]
            z = np.array([bbox[0],bbox[1]])
            kf = defineKalmanforme()
            kf.update(z)
            track = Track(bbox, features,kf)
            track.track_id = track_id_count
            track_id_count += 1
            tracks.append(track)
    

        
    if(len(tracks)>=1):
        tracks = Matched_tracks2 + Unmatched_track2 + New_detections2 
    
        # Display output
    for track in tracks:
        x, y, w, h = track.bbox
        color = colors[track.track_id % 100]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        # cv2.putText(frame, str(track.track_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord("q"):
        break
        
    # Release resources
cap.release()
cv2.destroyAllWindows()