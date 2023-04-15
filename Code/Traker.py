import numpy as np
import Track
# import kalman_filter
import filterpy
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class Traker:
    
    def __init__(self,max_age=60,max_dist = 0.7):
        self.tracks = []
        self.Next_id = 1
        # self.kf = KalmanFilter(dim_x=4,dim_z=2)
        self.max_age = max_age
        self.max_dist = max_dist
        
    
    def predict(self,unmatched_tracks):
        predicted_values = []
        for trac_ic in unmatched_tracks:
            predicted_values.append([self.tracks[trac_ic].predict(), self.tracks[trac_ic].xywh[0],self.tracks[trac_ic].xywh[1]])
        # return predicted_values
        
    def update(self,detections):
        if(len(self.tracks) == 0):
            ret_coo = []
            for d in detections:
                # print("Going Dark")
                self.addTrack(d)
                ret_coo.append(d.xywh)
            return ret_coo
        
        matches,matched_dets, unmatched_tracks, unmatched_dets = self._match(detections)
        
        
        tot = []
        matched_dd = []
        
        for trac_ic,det_ic in zip(matches,matched_dets):
            self.tracks[trac_ic].update(detections[det_ic].kf,detections[det_ic])
            tot.append(detections[det_ic].xywh)
            matched_dd.append(detections[det_ic].xywh)
            
        for det_ic in unmatched_dets:
            tot.append(detections[det_ic].xywh)
            self.addTrack(detections[det_ic])
        
        
        predicted_values = []
        for trac_ic in unmatched_tracks:
            self.tracks[trac_ic].num_missed()
            if (self.tracks[trac_ic].track_state != False):
                predicted_values.append(self.tracks[trac_ic].predict())
                v1,v2 = self.tracks[trac_ic].predict()
                if(v1!=0 and v2!=0):
                    tot.append([v1,v2,self.tracks[trac_ic].xywh[0],self.tracks[trac_ic].xywh[1]])
        
        for t in self.tracks:
            if t.track_state == False:
                self.tracks.remove(t)
            
        return tot
            
        
        
        
        
    def distance(self,feature1,feature2):
        return np.linalg.norm(feature1 - feature2)
        
    def _match(self,detections):
        
        # calculation of cost matrix
        cost_matrix = np.zeros((len(detections),len(self.tracks)))
        
        for i in range(len(detections)):
            for j in range(len(self.tracks)):
                # print("------------------")
                # print(detections[i].get_feature())
                # print(self.tracks[j].get_features())
                # print("------------------")
                A1 = detections[i].get_feature()
                A2 = self.tracks[j].get_features()
                cost_matrix[i][j] = self.distance(A1,A2)
        # for i,detection in enumerate(detections):
        #     for j,track in enumerate(self.tracks):
        #         cost_matrix[i][j] = self.distance(detection.get_feature(),track.get_features())
        
        
        print(cost_matrix)
        matched_tracks = []
        matched_dets = []
        unmatched_tracks = []
        unmatched_dets = []
        row_indices,column_indices = self.solve_Hungarian_algo(cost_matrix)
        
        
        for r,c in zip(row_indices,column_indices):
            if cost_matrix[r][c] < self.max_dist:
                # matched_tracks.append(self.tracks[c])
                # matched_dets.append(detections[r])
                matched_tracks.append(c)
                matched_dets.append(r)
                
            else:
                # unmatched_dets.append(detections[r])
                unmatched_dets.append(r)
                
        
        all_tracks_indx = []
        for i in range(len(self.tracks)):
            all_tracks_indx.append(i)
            
        all_tracks_indx = set(all_tracks_indx)    
        matched_tracks_indx = set(column_indices)
        
        unmatched_tracks_inds = all_tracks_indx - matched_tracks_indx
        for i in unmatched_tracks_inds:
            # unmatched_tracks.append(self.tracks[i])
            unmatched_tracks.append(i)
        
        
        return matched_tracks,matched_dets,unmatched_tracks,unmatched_dets
                
        
        
        
    def solve_Hungarian_algo(self,cost_matrix):
        row_indices,column_indices = linear_sum_assignment(cost_matrix)
        return row_indices,column_indices
    
    
    
    # xywh,score,next_id,feature,class_name
    def addTrack(self,detection):
        class_name = detection.get_class()
        xywh = detection.get_xywh()
        score = detection.get_score()
        feature = detection.get_feature()
        mean = detection.get_mean()
        covariance = detection.get_covariance()
        
        self.tracks.append(Track.Track(xywh,score,self.Next_id,feature,class_name,mean,covariance,self.max_age,track_state=True))
        
        self.Next_id += 1
        
        