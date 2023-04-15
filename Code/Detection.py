import numpy as np

class Detection:
    
    def __init__(self,xywh,score,feature,class_name,mean,covariance,kf) -> None:
        self.obj = {'xywh': xywh,
                    'score':score,
                    'feature':feature,
                    'class_name':class_name}
        
        self.xywh = xywh
        self.feature = feature
        self.score = score
        self.class_name = class_name
        self.mean = mean
        self.covariance = covariance
        self.kf = kf
        
        
    '''class_name = detection.get_class()
        xywh = detection.get_xywh()
        score = detection.get_score()
        feature = detection.get_feature()
        mean = detection.get_mean()
        covariance = detection.get_covariance()'''
        
    def get_class(self):
        return self.class_name
        
    def get_score(self):
        return self.score
    
    def get_feature(self):
        return self.feature 
    
    def get_xywh(self):
        return self.xywh
    
    def get_mean(self):
        return self.mean
    
    def get_covariance(self):
        return self.covariance