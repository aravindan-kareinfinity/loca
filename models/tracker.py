"""Person tracking using IoU + appearance features"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


class PersonTracker:
    """Custom tracking system using IoU + appearance features"""
    
    def __init__(self, max_age=30):
        self.tracks = {}
        self.next_track_id = 0
        self.max_age = max_age
        self.track_appearance_features = {}
        self.track_feature_extracted = {}
        self.track_side = {}  # Track which side of line each person is on

    @staticmethod
    def _iou(a, b):
        """Calculate Intersection over Union between two bounding boxes"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        xi1, yi1 = max(ax1, bx1), max(ay1, by1)
        xi2, yi2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _extract_appearance_features(self, frame, bbox):
        """Extract appearance features from bbox region (HSV color histograms)"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                return None
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return None
            roi_resized = cv2.resize(roi, (64, 128)) if roi.shape[0] > 0 and roi.shape[1] > 0 else roi
            hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            hist_h = hist_h / (np.sum(hist_h) + 1e-8)
            hist_s = hist_s / (np.sum(hist_s) + 1e-8)
            hist_v = hist_v / (np.sum(hist_v) + 1e-8)
            features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            return features
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a, b):
        """Calculate cosine similarity between two feature vectors"""
        if a is None or b is None:
            return 0.0
        try:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        except Exception:
            return 0.0

    def track(self, detections, frame=None):
        """Track detections using IoU + appearance features"""
        prev = self.tracks
        current = {}
        for tid in prev:
            prev[tid]['matched'] = False

        detection_features = []
        if frame is not None:
            for x1, y1, x2, y2, conf in detections:
                feat = self._extract_appearance_features(frame, (x1, y1, x2, y2))
                detection_features.append(feat)
        else:
            detection_features = [None] * len(detections)

        for idx, (x1, y1, x2, y2, conf) in enumerate(detections):
            bbox = (x1, y1, x2, y2)
            best_id = None
            best_score = -1.0
            det_feat = detection_features[idx] if idx < len(detection_features) else None

            for tid, tr in prev.items():
                iou = self._iou(bbox, tr['bbox'])
                score = iou
                if tid in self.track_appearance_features and det_feat is not None:
                    stored_feat = self.track_appearance_features[tid]
                    if stored_feat is not None:
                        feat_sim = self._cosine_similarity(det_feat, stored_feat)
                        score = 0.6 * iou + 0.4 * feat_sim
                if score > best_score and score > 0.3:
                    best_score = score
                    best_id = tid

            if best_id is not None:
                prev_track = prev[best_id]
                current[best_id] = {
                    'bbox': bbox,
                    'last_seen': 0,
                    'matched': True
                }
                if best_id in self.track_appearance_features:
                    if best_id not in self.track_feature_extracted or not self.track_feature_extracted.get(best_id, False):
                        if det_feat is not None:
                            self.track_appearance_features[best_id] = det_feat
                prev[best_id]['matched'] = True
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                current[tid] = {
                    'bbox': bbox,
                    'last_seen': 0,
                    'matched': True
                }
                if det_feat is not None:
                    self.track_appearance_features[tid] = det_feat
                    self.track_feature_extracted[tid] = False

        for tid, tr in prev.items():
            if not tr.get('matched', False):
                tr['last_seen'] += 1
                if tr['last_seen'] < self.max_age:
                    current[tid] = tr

        removed = set(prev.keys()) - set(current.keys())
        for rid in removed:
            self.track_appearance_features.pop(rid, None)
            self.track_feature_extracted.pop(rid, None)
            self.track_side.pop(rid, None)

        self.tracks = current
        return current

