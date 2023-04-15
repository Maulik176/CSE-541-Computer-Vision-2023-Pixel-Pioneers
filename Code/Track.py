import numpy as np
import kalman_filter

class Track:
    
    def __init__(self,xywh,score,next_id,feature,class_name,mean,covariance,max_track_age,track_state):
        self.obj = {'xywh': xywh,
                    'score':score,
                    'feature':feature,
                    'class_name':class_name}
        
        self.xywh = xywh
        self.feature = feature
        self.score = score
        self.class_name = class_name
        self.next_id = next_id
        self.mean = mean
        self.covariance = covariance
        self.track_age = 0
        self.max_age = max_track_age
        self.track_state = track_state
        self.kf_t = kalman_filter.KalmanF()
        
        
        
    def predict(self):
        # kf.predict()
        v1,v2 = self.kf_t.predict_() 
        return v1,v2
        
    def update(self,kf,detection):
        # self.mean = detection.get_mean()
        # self.covariance = detection.get_covariance()
        # z = np.array([x_center, y_center])
        z = np.array([detection.xywh[0],detection.xywh[1]])
        kf.update(z)
        self.track_age = 0
        
    def num_missed(self):
        # I have to predict the x,y for future
        # kf.predict()
        self.track_age+=1
        if self.track_age >= self.max_age:
            self.track_state = False
            # return 6942000,6942000
        
        # return kf.x[0],kf.x[1]
        
    def get_features(self):
        return self.feature
        