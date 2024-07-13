import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import os
import glob
import time
import argparse

class Detection:
    def __init__(self, bbox):
        self.bbox = np.array(bbox)

    @property
    def center(self):
        return (self.bbox[:2] + self.bbox[2:]) / 2

    @property
    def area(self):
        return np.prod(self.bbox[2:] - self.bbox[:2])

    @property
    def aspect_ratio(self):
        w, h = self.bbox[2:] - self.bbox[:2]
        return w / h

class Track:
    count = 0

    def __init__(self, detection):
        self.kf = self._init_kf(detection)
        self.id = Track.count
        Track.count += 1
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

    def _init_kf(self, detection):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        kf.R[2:,2:] *= 10
        kf.P[4:,4:] *= 1000
        kf.P *= 10
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01
        kf.x[:4] = self._detection_to_kf_state(detection)
        return kf

    def _detection_to_kf_state(self, detection):
        return np.r_[detection.center, detection.area, detection.aspect_ratio].reshape((4, 1))

    def predict(self):
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._kf_state_to_bbox()

    def update(self, detection):
        self.kf.update(self._detection_to_kf_state(detection))
        self.hits += 1
        self.time_since_update = 0

    def _kf_state_to_bbox(self):
        x, y, s, r = self.kf.x[:4].flatten()
        w = np.sqrt(s * r)
        h = s / w
        return np.array([x-w/2, y-h/2, x+w/2, y+h/2])

class SORT:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        
        # Predict new locations of tracks
        predicted_tracks = [track.predict() for track in self.tracks]
        
        # Match detections to tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections, predicted_tracks)
        
        # Update matched tracks
        for d, t in matched:
            self.tracks[t].update(detections[d])
        
        # Create new tracks
        for i in unmatched_dets:
            self.tracks.append(Track(detections[i]))
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Return active tracks
        return np.array([
            np.r_[t._kf_state_to_bbox(), t.id + 1]
            for t in self.tracks
            if t.hits >= self.min_hits or self.frame_count <= self.min_hits
        ])

    def _associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)
        
        iou_matrix = np.zeros((len(detections), len(trackers)))
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det.bbox, trk)
        
        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        unmatched_detections = [d for d in range(len(detections)) if d not in matched_indices[:, 0]]
        unmatched_trackers = [t for t in range(len(trackers)) if t not in matched_indices[:, 1]]
        
        matches = [
            (d, t) for d, t in matched_indices
            if iou_matrix[d, t] >= self.iou_threshold
        ]
        unmatched_detections.extend([d for d, t in matched_indices if iou_matrix[d, t] < self.iou_threshold])
        unmatched_trackers.extend([t for d, t in matched_indices if iou_matrix[d, t] < self.iou_threshold])
        
        return matches, unmatched_detections, unmatched_trackers

    @staticmethod
    def _iou(bbox1, bbox2):
        bbox1 = np.expand_dims(bbox1, 0)
        bbox2 = np.expand_dims(bbox2, 1)
        
        xx1 = np.maximum(bbox1[..., 0], bbox2[..., 0])
        yy1 = np.maximum(bbox1[..., 1], bbox2[..., 1])
        xx2 = np.minimum(bbox1[..., 2], bbox2[..., 2])
        yy2 = np.minimum(bbox1[..., 3], bbox2[..., 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])                                      
            + (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1]) - wh)                                              
        return o

def parse_args():
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', action='store_true', help='Display online tracker output (slow)')
    parser.add_argument('--seq_path', type=str, default='data', help='Path to detections')
    parser.add_argument('--phase', type=str, default='train', help='Subdirectory in seq_path')
    parser.add_argument('--max_age', type=int, default=1, help='Maximum frames to keep alive a track without associated detections')
    parser.add_argument('--min_hits', type=int, default=3, help='Minimum hits to start track')
    parser.add_argument('--iou_threshold', type=float, default=0.3, help='Minimum IOU for match')
    return parser.parse_args()

def main():
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)

    if display:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = SORT(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
        
        with open(os.path.join('output', f'{seq}.txt'), 'w') as out_file:
            print(f"Processing {seq}.")
            for frame in range(int(seq_dets[:,0].max())):
                frame += 1
                dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2]
                total_frames += 1

                if display:
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', f'{frame:06d}.jpg')
                    im = io.imread(fn)
                    ax1.imshow(im)
                    plt.title(f'{seq} Tracked Targets')

                start_time = time.time()
                trackers = mot_tracker.update([Detection(det) for det in dets])
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print(f'{frame},{d[4]:.0f},{d[0]:.2f},{d[1]:.2f},{d[2]-d[0]:.2f},{d[3]-d[1]:.2f},1,-1,-1,-1', file=out_file)
                    if display:
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

                if display:
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()

    print(f"Total Tracking took: {total_time:.3f} seconds for {total_frames} frames or {total_frames / total_time:.1f} FPS")

    if display:
        print("Note: to get real runtime results run without the option: --display")

if __name__ == '__main__':
    main()