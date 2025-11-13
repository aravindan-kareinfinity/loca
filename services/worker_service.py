"""Camera worker service for processing video streams"""

import threading
import time
import subprocess
import numpy as np
import cv2
from collections import deque
from typing import Dict, Optional

# Handle both relative and absolute imports
try:
    from ..models.camera import Camera
    from ..models.tracker import PersonTracker
    from ..models.counter import PersonCounter
    from ..services.count_log_service import CountLogService
except ImportError:
    from models.camera import Camera
    from models.tracker import PersonTracker
    from models.counter import PersonCounter
    from services.count_log_service import CountLogService

# FFmpeg frame reading configuration
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
BYTES_PER_FRAME = FRAME_WIDTH * FRAME_HEIGHT * 3


class FrameBuffer:
    """Thread-safe frame buffer"""
    
    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.frame_available = threading.Condition(self.lock)
    
    def put(self, frame):
        """Add a frame to the buffer"""
        with self.lock:
            self.buffer.append(frame)
            self.frame_available.notify()
    
    def get(self, timeout=None):
        """Get a frame from the buffer"""
        with self.frame_available:
            if not self.buffer:
                self.frame_available.wait(timeout)
            if self.buffer:
                return self.buffer.popleft()
            return None
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()
    
    def size(self):
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)


def start_ffmpeg(rtsp_url, show_errors=False):
    """Start FFmpeg process for RTSP stream"""
    try:
        cmd = [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-timeout", "5000000",
            "-i", rtsp_url,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
            "-r", "15",
            "-"
        ]
        stderr_output = None if show_errors else subprocess.DEVNULL
        pipe = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=stderr_output,
            bufsize=10 * BYTES_PER_FRAME
        )
        
        time.sleep(0.5)  # Give FFmpeg time to initialize
        
        if pipe.poll() is not None:
            return None
        
        return pipe
    except FileNotFoundError:
        return None
    except Exception:
        return None


def safe_terminate(pipe):
    """Safely terminate FFmpeg process"""
    if pipe is not None:
        try:
            pipe.terminate()
            pipe.wait(timeout=2)
        except:
            try:
                pipe.kill()
            except:
                pass


def frame_reader_thread(pipe, frame_buffer, stop_event):
    """Thread function to read frames from FFmpeg"""
    partial_data = b""
    
    while not stop_event.is_set():
        try:
            if pipe.poll() is not None:
                break
            
            chunk = pipe.stdout.read(BYTES_PER_FRAME // 4)
            if not chunk:
                time.sleep(0.01)
                continue
            
            partial_data += chunk
            
            while len(partial_data) >= BYTES_PER_FRAME:
                frame_data = partial_data[:BYTES_PER_FRAME]
                partial_data = partial_data[BYTES_PER_FRAME:]
                
                try:
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(
                        (FRAME_HEIGHT, FRAME_WIDTH, 3)
                    ).copy()
                    
                    if frame_buffer.size() < 5:
                        frame_buffer.put(frame)
                except Exception:
                    continue
                    
        except Exception:
            break


class CameraWorker:
    """Worker for processing camera streams"""
    
    def __init__(self, cam: Camera, count_log_service: Optional[CountLogService] = None):
        self.cam = cam
        self.thread = None
        self.reader_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.ffmpeg_pipe = None
        self.frame_buffer = FrameBuffer(max_size=5)
        self.count_log_service = count_log_service
        
        # Person detection and tracking
        self.tracker = PersonTracker(max_age=30)
        self.person_counter = None
        self.count_in = 0
        self.count_out = 0
        self.frame_counter = 0
        self.prev_frame = None
        self.prev_processed_frame = None
        self._init_counter()
    
    def _init_counter(self):
        """Initialize person counter with tripwire line settings"""
        if self.cam.lineEnabled and self.cam.lineX1 is not None and self.cam.lineY1 is not None and \
           self.cam.lineX2 is not None and self.cam.lineY2 is not None:
            x1 = int(self.cam.lineX1 * (FRAME_WIDTH / 640) if 640 > 0 else 1.0)
            y1 = int(self.cam.lineY1 * (FRAME_HEIGHT / 360) if 360 > 0 else 1.0)
            x2 = int(self.cam.lineX2 * (FRAME_WIDTH / 640) if 640 > 0 else 1.0)
            y2 = int(self.cam.lineY2 * (FRAME_HEIGHT / 360) if 360 > 0 else 1.0)
            self.person_counter = PersonCounter(
                x1, y1, x2, y2,
                direction=self.cam.lineDirection,
                count_mode=self.cam.lineCountMode,
                camera_id=self.cam.idCode
            )
            if hasattr(self, 'person_counter') and self.person_counter is not None:
                self.person_counter.count_in = self.count_in
                self.person_counter.count_out = self.count_out
    
    def start(self, force_restart=False):
        """Start the worker"""
        if self.thread and self.thread.is_alive() and not force_restart:
            return
        if force_restart and self.thread and self.thread.is_alive():
            self.stop_event.set()
            self.thread.join(timeout=2)
            if self.reader_thread and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=2)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the worker"""
        self.stop_event.set()
        if self.reader_thread and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        if self.ffmpeg_pipe:
            safe_terminate(self.ffmpeg_pipe)
            self.ffmpeg_pipe = None
    
    def _encode_jpeg(self, frame):
        """Encode frame as JPEG"""
        if cv2 is None:
            return None
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            return None
        return buf.tobytes()
    
    def _run(self):
        """Main worker loop"""
        if cv2 is None:
            while not self.stop_event.is_set():
                time.sleep(0.5)
            return
        
        url = self.cam.connection_url()
        if not url:
            return
        
        # Start FFmpeg process
        self.ffmpeg_pipe = start_ffmpeg(url, show_errors=False)
        if self.ffmpeg_pipe is None:
            return
        
        # Start frame reader thread
        self.reader_thread = threading.Thread(
            target=frame_reader_thread,
            args=(self.ffmpeg_pipe, self.frame_buffer, self.stop_event),
            daemon=True
        )
        self.reader_thread.start()
        
        # Main loop - get frames from buffer and encode
        while not self.stop_event.is_set():
            frame = self.frame_buffer.get(timeout=0.1)
            if frame is None:
                # Check if FFmpeg process died
                if self.ffmpeg_pipe and self.ffmpeg_pipe.poll() is not None:
                    safe_terminate(self.ffmpeg_pipe)
                    time.sleep(2)
                    self.ffmpeg_pipe = start_ffmpeg(url, show_errors=False)
                    if self.ffmpeg_pipe:
                        self.reader_thread = threading.Thread(
                            target=frame_reader_thread,
                            args=(self.ffmpeg_pipe, self.frame_buffer, self.stop_event),
                            daemon=True
                        )
                        self.reader_thread.start()
                        self.frame_buffer.clear()
                continue
            
            # Process frame: detect, track, and count people
            frame = self._process_frame(frame)
            
            # Encode and store frame
            jpeg = self._encode_jpeg(frame)
            if jpeg is not None:
                with self.lock:
                    self.latest_jpeg = jpeg
            time.sleep(0.03)
        
        # Cleanup
        if self.ffmpeg_pipe:
            safe_terminate(self.ffmpeg_pipe)
            self.ffmpeg_pipe = None
    
    def _process_frame(self, frame):
        """Detect people, track them, and count crossings"""
        # Import YOLO model from utils - will be injected by worker manager
        # For now, we'll need to pass it or make it a module-level import
        try:
            try:
                from ..utils.yolo_loader import get_yolo_model
            except ImportError:
                from utils.yolo_loader import get_yolo_model
            import os
            import sys
            if getattr(sys, 'frozen', False):
                project_dir = os.path.dirname(os.path.abspath(sys.executable))
            else:
                project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            yolo_model = get_yolo_model(project_dir=project_dir)
        except Exception:
            yolo_model = None
        
        if yolo_model is None or cv2 is None:
            return frame
        
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # Draw tripwire line if enabled
            frame_has_changes = True
            if self.prev_frame is not None:
                gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(gray_current, gray_prev)
                change_amount = np.sum(frame_diff) / (frame_width * frame_height)
                
                if change_amount < 0.5:
                    frame_has_changes = False
            
            # Draw tripwire line if enabled
            if self.cam.lineEnabled and self.cam.lineX1 is not None and self.cam.lineY1 is not None and \
               self.cam.lineX2 is not None and self.cam.lineY2 is not None:
                try:
                    scale_x = frame_width / FRAME_WIDTH if FRAME_WIDTH > 0 else 1.0
                    scale_y = frame_height / FRAME_HEIGHT if FRAME_HEIGHT > 0 else 1.0
                    x1 = int(self.cam.lineX1 * scale_x)
                    y1 = int(self.cam.lineY1 * scale_y)
                    x2 = int(self.cam.lineX2 * scale_x)
                    y2 = int(self.cam.lineX2 * scale_y)
                    
                    # Update counter if line changed
                    if self.person_counter is None or \
                       self.person_counter.line_x1 != x1 or self.person_counter.line_y1 != y1 or \
                       self.person_counter.line_x2 != x2 or self.person_counter.line_y2 != y2:
                        self.person_counter = PersonCounter(
                            x1, y1, x2, y2,
                            direction=self.cam.lineDirection,
                            count_mode=self.cam.lineCountMode,
                            camera_id=self.cam.idCode
                        )
                        self.person_counter.count_in = self.count_in
                        self.person_counter.count_out = self.count_out
                    
                    # Draw elegant thin tripwire line
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 1)
                    
                    # Draw IN/OUT labels
                    count_mode = self.cam.lineCountMode.upper() if hasattr(self.cam, 'lineCountMode') else "BOTH"
                    mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    vx, vy = (x2 - x1), (y2 - y1)
                    nx, ny = -vy, vx
                    norm = (nx**2 + ny**2) ** 0.5 or 1.0
                    nx, ny = nx / norm, ny / norm
                    offset = 25
                    in_left = (self.cam.lineDirection == "AtoBIsIn")
                    in_pos = (int(mx + (nx if in_left else -nx) * offset), int(my + (ny if in_left else -ny) * offset))
                    out_pos = (int(mx + (-nx if in_left else nx) * offset), int(my + (-ny if in_left else ny) * offset))
                    
                    if count_mode == "BOTH":
                        cv2.putText(frame, "IN", in_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, "OUT", out_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    elif count_mode == "IN":
                        cv2.putText(frame, "IN", in_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    elif count_mode == "OUT":
                        cv2.putText(frame, "OUT", out_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception:
                    pass
            
            # Store current frame as previous
            self.prev_frame = frame.copy()
            
            # Process only alternative frames
            should_process = (self.frame_counter % 2 == 0)
            
            if should_process:
                # Run YOLO detection
                results = yolo_model(frame, classes=[0], conf=0.5, verbose=False)
                detections = []
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            cls = int(box.cls[0].cpu().numpy())
                            if cls == 0:  # person
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
                                detections.append((float(x1), float(y1), float(x2), float(y2), conf))
                
                # Track people
                tracks = self.tracker.track(detections, frame=frame)
            else:
                # Age existing tracks
                for tid in list(self.tracker.tracks.keys()):
                    track = self.tracker.tracks[tid]
                    track['last_seen'] += 1
                    if track['last_seen'] >= self.tracker.max_age:
                        del self.tracker.tracks[tid]
                        self.tracker.track_appearance_features.pop(tid, None)
                        self.tracker.track_feature_extracted.pop(tid, None)
                        self.tracker.track_side.pop(tid, None)
                tracks = self.tracker.tracks
            
            # Increment frame counter
            self.frame_counter += 1
            
            # Process each track for counting
            for track_id, track_info in tracks.items():
                bbox = track_info['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                
                # Calculate center point (bottom center for feet position)
                center_x = (x1 + x2) // 2
                center_y = y2  # Bottom of bounding box
                
                # Check crossing if counter is initialized
                if self.person_counter is not None:
                    crossed, direction = self.person_counter.check_crossing(
                        track_id, center_x, center_y, self.tracker.track_side
                    )
                    if crossed:
                        with self.lock:
                            self.count_in = self.person_counter.count_in
                            self.count_out = self.person_counter.count_out
                        
                        # Log the event
                        if self.count_log_service:
                            self.count_log_service.append_count_event(
                                self.cam.idCode, direction, self.count_in, self.count_out
                            )
                    
                    # Color based on crossing
                    if crossed:
                        box_color = (0, 255, 0) if direction == "in" else (0, 0, 255)
                        thickness = 2
                    else:
                        box_color = (255, 255, 0)
                        thickness = 1
                else:
                    box_color = (255, 255, 0)
                    thickness = 1
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
                cv2.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
            
        except Exception:
            pass
        
        # Store processed frame
        if frame is not None:
            self.prev_processed_frame = frame.copy()
        
        return frame
    
    def get_jpeg(self):
        """Get latest JPEG frame"""
        with self.lock:
            return self.latest_jpeg
    
    def get_counts(self):
        """Get current IN/OUT counts"""
        with self.lock:
            counts = {
                'count_in': self.count_in,
                'count_out': self.count_out,
                'count_inside': max(0, self.count_in - self.count_out)
            }
            return counts


class WorkerManager:
    """Manager for camera workers"""
    
    def __init__(self, count_log_service: Optional[CountLogService] = None):
        self.workers: Dict[str, CameraWorker] = {}
        self.workers_lock = threading.Lock()
        self.count_log_service = count_log_service
    
    def start_worker_for_camera(self, cam: Camera):
        """Start a worker for a camera"""
        with self.workers_lock:
            worker = self.workers.get(cam.idCode)
            if worker is None:
                worker = CameraWorker(cam, self.count_log_service)
                self.workers[cam.idCode] = worker
            else:
                worker.cam = cam
                worker._init_counter()
        worker.start()
    
    def stop_worker(self, id_code: str):
        """Stop a worker"""
        worker = None
        with self.workers_lock:
            worker = self.workers.get(id_code)
        if worker:
            worker.stop()
            with self.workers_lock:
                self.workers.pop(id_code, None)
    
    def get_worker(self, id_code: str) -> Optional[CameraWorker]:
        """Get a worker by ID"""
        with self.workers_lock:
            return self.workers.get(id_code)
    
    def get_all_workers(self) -> Dict[str, CameraWorker]:
        """Get all workers"""
        with self.workers_lock:
            return dict(self.workers)

