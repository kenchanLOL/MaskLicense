from __future__ import absolute_import
from deep_sort.tracker import Tracker
import numpy as np
from numpy.core.numeric import indices
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
from deep_sort import preprocessing


class simipleTracker(Tracker):
    def __init__(self,max_age=120, n_init=2):
        self.max_age=max_age
        self.n_init = n_init
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = {}
        self._next_id = 1
    def predict(self):
        for track_id in self.tracks:
            self.tracks[track_id].predict(self.kf)
    def update(self, detections):
        for detect_id in detections:
            if detect_id in self.tracks:
                self.tracks[detect_id].update(
                    self.kf,detections[detect_id])
            else:
                self._initiate_track(detections[detect_id])
    def _initiate_track(self, detection):
        bbox=detection[:4]
        height=detection[3]-detection[1]
        width=detection[2]-detection[0]
        bbox[:2]+=bbox[2:]/2
        bbox[2]=width/height
        bbox[3]=height
        mean, covariance = self.kf.initiate(detection)
        self.tracks.append(Track(mean,covariance,self._next_id,self.n_init,self.max_age))
        self._next_id+=1       