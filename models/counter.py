"""Tripwire counting using point-side logic"""

import numpy as np
from typing import Dict, Optional, Tuple


class PersonCounter:
    """Tripwire counting using point-side logic"""
    
    def __init__(self, line_x1, line_y1, line_x2, line_y2, direction="AtoBIsIn", count_mode="BOTH", camera_id=None):
        self.line_x1 = line_x1
        self.line_y1 = line_y1
        self.line_x2 = line_x2
        self.line_y2 = line_y2
        self.direction = direction
        self.count_mode = count_mode
        self.count_in = 0
        self.count_out = 0
        self.camera_id = camera_id  # Store camera ID for logging

    @staticmethod
    def _point_side(px, py, x1, y1, x2, y2):
        """Returns signed area factor: >0 left of line A->B, <0 right, 0 on line"""
        return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

    def check_crossing(self, track_id, center_x, center_y, track_side_dict):
        """Check if person crossed the line using point-side logic"""
        side = self._point_side(center_x, center_y, self.line_x1, self.line_y1, self.line_x2, self.line_y2)
        prev_side = track_side_dict.get(track_id)
        track_side_dict[track_id] = side

        crossed = False
        direction = None

        if prev_side is not None and prev_side != 0 and side != 0 and np.sign(prev_side) != np.sign(side):
            # Determine crossing direction
            # prev_side < 0 means was on RIGHT side of line A->B
            # prev_side > 0 means was on LEFT side of line A->B
            # side > 0 means now on LEFT side of line A->B
            # side < 0 means now on RIGHT side of line A->B

            # Crossing from RIGHT to LEFT: prev_side < 0 -> side > 0 (moving from B side to A side = B->A)
            # Crossing from LEFT to RIGHT: prev_side > 0 -> side < 0 (moving from A side to B side = A->B)
            crossed_B_to_A = (prev_side < 0 and side > 0)  # Right to left = B->A
            crossed_A_to_B = (prev_side > 0 and side < 0)  # Left to right = A->B

            # Determine event based on direction setting
            # AtoBIsIn: A->B crossing is IN, B->A crossing is OUT
            # BtoAIsIn: B->A crossing is IN, A->B crossing is OUT
            if self.direction == 'AtoBIsIn':
                if crossed_A_to_B:
                    event = 'in'
                else:  # crossed_B_to_A
                    event = 'out'
            else:  # BtoAIsIn
                if crossed_B_to_A:
                    event = 'in'
                else:  # crossed_A_to_B
                    event = 'out'

            # Check if count_mode filters this event
            if (self.count_mode == 'IN' and event != 'in') or \
               (self.count_mode == 'OUT' and event != 'out'):
                return False, None

            # Increment the appropriate counter
            if event == 'in':
                self.count_in += 1
                crossed, direction = True, "in"
            else:  # event == 'out'
                self.count_out += 1
                crossed, direction = True, "out"

        return crossed, direction

