import filterpy
from filterpy.kalman import KalmanFilter
import numpy as np

class KalmanF:
    
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4,dim_z=2)
        self.kf.x = np.zeros(4)  # Initial state (x, y, vx, vy)
        self.kf.P = np.eye(4) * 1000  # Initial state covariance
        self.kf.F = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])  # State transition matrix
        self.kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])  # Measurement matrix
        self.kf.R = np.eye(2) * 10  # Measurement noise covariance
        self.kf.Q = np.eye(4) * 0.1
        
    
    
    def update(self,z):
        self.kf.update(z)
        
    def predict_(self):
        self.kf.predict()
        return self.kf.x[0] , self.kf.x[1]