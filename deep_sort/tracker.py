# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from numpy.core.numeric import indices
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

from deep_sort import preprocessing

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.6, max_age=120, n_init=2):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """

        # Run matching cascade.
        self.matches, self.unmatched_tracks, self.unmatched_detections = \
            self._match(detections)
        # print(unmatched_detections)
        # Update track set.
        # print('tracks:')
        # print(self.tracks)
        print('matches tracks:')
        print(self.matches)
        print('unmatched tracks:')
        print(self.unmatched_tracks)
        print('unmatched detections:')
        print(self.unmatched_detections)
        for track_idx, detection_idx in self.matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in self.unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in self.unmatched_detections:
            self.matches.append([self._next_id,detection_idx])
            self._initiate_track(detections[detection_idx])            
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        temp=[t.to_tlbr() for t in self.tracks]
        indices = preprocessing.non_max_suppression(np.array(temp), classes=None, max_bbox_overlap=0.7 ,scores=None)
        for t in self.tracks:
            if(t.is_tentative()):
                continue
            elif not (t.track_id in indices):
                t.time_since_update+=1

        
        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        print("========================================================")
        print("confirmed_track:")
        for k in confirmed_tracks:
            print(self.tracks[k].to_tlbr())
        print("unconfirmed_tracks:")
        for k in unconfirmed_tracks:
            print(self.tracks[k].to_tlbr())
        print("========================================================")

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name = detection.get_class()
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, class_name))
        self._next_id += 1

