from flask import Flask, send_from_directory, request, jsonify
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from datetime import datetime
import uuid
import threading
import time
from typing import Dict

try:
    import cv2
except Exception:
    cv2 = None
try:
    import numpy as np
except Exception:
    np = None
try:
    import pickle
except Exception:
    pickle = None
try:
    import face_recognition
except Exception:
    face_recognition = None
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# GPU acceleration imports
try:
    import torch
    import torch.backends.cudnn as cudnn
except Exception:
    torch = None

ProtocolType = Literal["rtsp", "onvif", "http", "https", "rtmp"]
ModeType = Literal["Recognize", "Identify", "Recognize and Identify"]


@dataclass
class Camera:
    idCode: str
    name: str
    location: str
    description: Optional[str] = None
    ipOrHost: str = ""
    port: Optional[int] = None
    protocol: ProtocolType = "rtsp"
    streamPath: Optional[str] = None
    username: Optional[str] = None
    passwordEnc: Optional[str] = None
    macAddress: Optional[str] = None
    mode: ModeType = "Recognize"
    active: bool = True
    createdAt: datetime = field(default_factory=datetime.utcnow)
    updatedAt: datetime = field(default_factory=datetime.utcnow)
    lineEnabled: bool = False
    lineX1: Optional[int] = None
    lineY1: Optional[int] = None
    lineX2: Optional[int] = None
    lineY2: Optional[int] = None
    lineDirection: str = "AtoBIsIn"
    lineCountMode: str = "BOTH"

    @property
    def streamingUrl(self) -> str:
        return self.connection_url()

    def connection_url(self) -> str:
        host = (self.ipOrHost or "").strip()
        if not host:
            return ""
        proto = (self.protocol or "rtsp").lower().rstrip(":")
        auth = ""
        if self.username:
            auth = self.username
            if self.passwordEnc:
                auth = f"{auth}:{self.passwordEnc}"
            auth += "@"
        port = f":{self.port}" if self.port else ""
        path = f"/{(self.streamPath or '').lstrip('/')}" if self.streamPath else ""
        return f"{proto}://{auth}{host}{port}{path}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["streamingUrl"] = self.streamingUrl
        data["createdAt"] = self.createdAt.isoformat()
        data["updatedAt"] = self.updatedAt.isoformat()
        return data


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = BASE_DIR
FACE_DB_PATH = os.path.join(PROJECT_DIR, "face_database.pkl")
PRESENCE_LOG_PATH = os.path.join(PROJECT_DIR, "presence_log.json")
PERSON_IMAGES_DIR = os.path.join(PROJECT_DIR, "person_images")
YOLO_WEIGHTS = os.path.join(PROJECT_DIR, "yolov8n.pt")

app = Flask(__name__, static_folder=PROJECT_DIR, static_url_path="/")

# GPU Configuration
class GPUConfig:
    def __init__(self):
        self.use_gpu = False
        self.gpu_device = "cuda:0"
        self.available_gpus = 0
        self.setup_gpu()
    
    def setup_gpu(self):
        """Initialize GPU settings"""
        if torch is not None and torch.cuda.is_available():
            self.available_gpus = torch.cuda.device_count()
            self.use_gpu = True
            cudnn.benchmark = True  # Optimize for fixed input sizes
            print(f"✅ GPU Acceleration Enabled: {self.available_gpus} GPU(s) available")
            for i in range(self.available_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ GPU not available - falling back to CPU")
            self.use_gpu = False
    
    def get_device(self, gpu_id=0):
        """Get appropriate device (GPU/CPU)"""
        if self.use_gpu and gpu_id < self.available_gpus:
            return f"cuda:{gpu_id}"
        return "cpu"

gpu_config = GPUConfig()


class CameraWorker:
    def __init__(self, cam: 'Camera'):
        self.cam = cam
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.mode = cam.mode
        self.recognizer = None
        self.line_config = {
            "enabled": bool(getattr(cam, 'lineEnabled', False)),
            "x1": getattr(cam, 'lineX1', None),
            "y1": getattr(cam, 'lineY1', None),
            "x2": getattr(cam, 'lineX2', None),
            "y2": getattr(cam, 'lineY2', None),
            "direction": getattr(cam, 'lineDirection', 'AtoBIsIn'),
            "countMode": getattr(cam, 'lineCountMode', 'BOTH'),
            "cameraId": cam.idCode
        }

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

    def _encode_jpeg(self, frame):
        if cv2 is None:
            return None
        ok, buf = cv2.imencode('.jpg', frame)
        if not ok:
            return None
        return buf.tobytes()

    def _process_recognize(self, frame):
        if cv2 is None:
            return frame
        self._ensure_recognizer()
        if self.recognizer:
            self.recognizer.process_frame(frame, self.line_config)
        cv2.putText(frame, 'Recognize mode', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        return frame

    def _process_detect_train(self, frame):
        if cv2 is None:
            return frame
        self._ensure_recognizer()
        if self.recognizer:
            self.recognizer.process_frame(frame, self.line_config)
        cv2.putText(frame, 'Detect/Train mode', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        return frame

    def _process_full(self, frame):
        if cv2 is None:
            return frame
        self._ensure_recognizer()
        if self.recognizer:
            self.recognizer.process_frame(frame, self.line_config)
        cv2.putText(frame, 'Full mode', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        return frame

    def _ensure_recognizer(self):
        if self.recognizer is not None:
            return
        if cv2 is None or face_recognition is None or np is None:
            return
        try:
            self.recognizer = FaceRecognitionSystem()
        except Exception:
            self.recognizer = None

    def _run(self):
        if cv2 is None:
            while not self.stop_event.is_set():
                time.sleep(0.5)
            return
        
        # Try to use GPU-accelerated video capture if available
        url = self.cam.connection_url()
        try:
            # Try using OpenCV with GPU support
            cap = cv2.VideoCapture(url)
            # Set hardware acceleration if available
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, 1)  # Enable hardware acceleration
        except:
            cap = cv2.VideoCapture(url)
            
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Frame skipping for performance
        frame_skip_counter = 0
        frame_skip_interval = 1  # Process every 2nd frame
        
        while not self.stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue
                
            # Frame skipping to reduce load
            frame_skip_counter += 1
            if frame_skip_counter % (frame_skip_interval + 1) != 0:
                continue
                
            # Reduce resolution for faster processing
            if frame.shape[1] > 1280:  # If width > 1280, resize
                frame = cv2.resize(frame, (1280, 720))
                
            mode = (self.mode or 'Recognize')
            if mode == 'Recognize':
                frame = self._process_recognize(frame)
            elif mode == 'Identify':
                frame = self._process_detect_train(frame)
            else:
                frame = self._process_full(frame)

            jpeg = self._encode_jpeg(frame)
            if jpeg is not None:
                with self.lock:
                    self.latest_jpeg = jpeg
                    
            time.sleep(0.03)
        cap.release()

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg


workers: Dict[str, CameraWorker] = {}

def start_worker_for_camera(cam: 'Camera'):
    worker = workers.get(cam.idCode)
    if worker is None:
        worker = CameraWorker(cam)
        workers[cam.idCode] = worker
    worker.mode = cam.mode
    worker.line_config = {
        "enabled": bool(getattr(cam, 'lineEnabled', False)),
        "x1": getattr(cam, 'lineX1', None),
        "y1": getattr(cam, 'lineY1', None),
        "x2": getattr(cam, 'lineX2', None),
        "y2": getattr(cam, 'lineY2', None),
        "direction": getattr(cam, 'lineDirection', 'AtoBIsIn'),
        "countMode": getattr(cam, 'lineCountMode', 'BOTH'),
        "cameraId": cam.idCode
    }
    worker.start()

def stop_worker(id_code: str):
    worker = workers.get(id_code)
    if worker:
        worker.stop()
        workers.pop(id_code, None)


def start_webcam_worker():
    cam = Camera(
        idCode="WEBCAM",
        name="Webcam",
        location="Local",
        ipOrHost="",
        protocol="rtsp",
        streamPath=None,
        active=True,
        mode="Recognize"
    )
    worker = workers.get("WEBCAM")
    if worker is None:
        worker = CameraWorker(cam)
        workers["WEBCAM"] = worker
    worker.mode = cam.mode
    
    def run_webcam():
        if cv2 is None:
            while not worker.stop_event.is_set():
                time.sleep(0.5)
            return
            
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        worker._ensure_recognizer()
        
        frame_skip_counter = 0
        frame_skip_interval = 1
        
        while not worker.stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
                
            frame_skip_counter += 1
            if frame_skip_counter % (frame_skip_interval + 1) != 0:
                continue
                
            if frame.shape[1] > 1280:
                frame = cv2.resize(frame, (1280, 720))
                
            mode = (worker.mode or 'Recognize')
            if mode == 'Recognize':
                frame = worker._process_recognize(frame)
            elif mode == 'Identify':
                frame = worker._process_detect_train(frame)
            else:
                frame = worker._process_full(frame)
                
            jpeg = worker._encode_jpeg(frame)
            if jpeg is not None:
                with worker.lock:
                    worker.latest_jpeg = jpeg
            time.sleep(0.03)
        cap.release()
        
    worker.stop_event.clear()
    worker.thread = threading.Thread(target=run_webcam, daemon=True)
    worker.thread.start()


@app.get("/")
def serve_root():
    return send_from_directory(PROJECT_DIR, "index.html")


@app.get("/<path:path>")
def serve_assets(path: str):
    return send_from_directory(PROJECT_DIR, path)


@app.get('/api/stream/<id_code>')
def stream_camera(id_code: str):
    from flask import Response
    worker = workers.get(id_code)
    if worker is None:
        return jsonify({"error": "stream not available"}), 404

    def gen():
        boundary = b'--frame\r\n'
        while True:
            jpeg = worker.get_jpeg()
            if jpeg is not None:
                yield boundary
                yield b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n'
            time.sleep(0.03)

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.unknown_faces_buffer = {}
        self.person_id_to_name = {}
        self.face_encoding_cache = {}
        self.creation_cooldown = {}
        self.next_track_id = 0
        self.tracks = {}
        self.max_age = 2
        self.next_person_id = 1
        self.active_person_counts = {}
        self.track_side = {}
        self.track_counted_in = {}
        
        # GPU device
        self.device = gpu_config.get_device()

        os.makedirs(PERSON_IMAGES_DIR, exist_ok=True)
        self._load_face_database()
        self.presence_events = self._load_presence_log()

        # YOLO person detector with GPU
        self.yolo_model = None
        if YOLO is not None and os.path.exists(YOLO_WEIGHTS):
            try:
                self.yolo_model = YOLO(YOLO_WEIGHTS)
                # Move YOLO to GPU if available
                if gpu_config.use_gpu:
                    self.yolo_model.to(self.device)
                    print(f"✅ YOLO model loaded on {self.device}")
            except Exception as e:
                print(f"❌ YOLO initialization failed: {e}")
                self.yolo_model = None

        # Initialize mapping for known faces
        for i, name in enumerate(self.known_face_names):
            person_id = i + 1
            self.person_id_to_name[person_id] = name
            if i < len(self.known_face_encodings):
                self.face_encoding_cache[person_id] = self.known_face_encodings[i]
        self.next_person_id = len(self.known_face_names) + 1

    def _load_face_database(self):
        if pickle is None:
            return
        if os.path.exists(FACE_DB_PATH):
            try:
                with open(FACE_DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get('encodings', [])
                    self.known_face_names = data.get('names', [])
            except Exception:
                self.known_face_encodings = []
                self.known_face_names = []

    def _save_face_database(self):
        if pickle is None:
            return
        try:
            with open(FACE_DB_PATH, 'wb') as f:
                pickle.dump({'encodings': self.known_face_encodings, 'names': self.known_face_names}, f)
        except Exception:
            pass

    def reload_face_database_names(self):
        if pickle is None:
            return
        try:
            if os.path.exists(FACE_DB_PATH):
                with open(FACE_DB_PATH, 'rb') as f:
                    data = pickle.load(f)
                    names = data.get('names', [])
                    if isinstance(names, list) and len(names) == len(self.known_face_encodings):
                        self.known_face_names = names
                        self.person_id_to_name = {i + 1: nm for i, nm in enumerate(self.known_face_names)}
        except Exception:
            pass

    def _load_presence_log(self):
        try:
            if os.path.exists(PRESENCE_LOG_PATH):
                with open(PRESENCE_LOG_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
        except Exception:
            return []
        return []

    def _save_presence_log(self):
        try:
            with open(PRESENCE_LOG_PATH, 'w', encoding='utf-8') as f:
                json.dump(self.presence_events, f, indent=2)
        except Exception:
            pass

    def _append_presence_event(self, person_id, name, event, image_path=None, camera_id=None):
        record = {
            "timestamp": datetime.now().isoformat(timespec='seconds'),
            "person_id": person_id,
            "name": name,
            "event": event
        }
        if image_path:
            record["image_path"] = os.path.relpath(image_path, PROJECT_DIR)
        if camera_id:
            record["camera_id"] = camera_id
        self.presence_events.append(record)
        self._save_presence_log()

    def _ensure_person_folder(self, person_id):
        folder = os.path.join(PERSON_IMAGES_DIR, f"person_{person_id}")
        os.makedirs(folder, exist_ok=True)
        return folder

    def _save_person_image(self, person_id, face_image):
        try:
            folder = self._ensure_person_folder(person_id)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(folder, f"{ts}.jpg")
            cv2.imwrite(path, face_image)
            return path
        except Exception:
            return None

    def _detect_people(self, frame):
        # Use YOLO with GPU acceleration
        if self.yolo_model is not None:
            try:
                # YOLO automatically uses GPU if model is on GPU
                results = self.yolo_model(frame, classes=[0], verbose=False, conf=0.5, device=self.device)
                detections = []
                for result in results:
                    if getattr(result, 'boxes', None) is not None:
                        for box in result.boxes:
                            cls = int(box.cls[0].cpu().numpy())
                            if cls == 0:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
                                detections.append([x1, y1, x2, y2, conf])
                return detections
            except Exception as e:
                print(f"YOLO detection failed: {e}")
                pass
                
        # Fallback to CPU face detection
        if face_recognition is None:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb, model="hog")
        detections = []
        for top, right, bottom, left in face_locations:
            h = bottom - top
            w = right - left
            x1 = max(0, left - w//2)
            y1 = max(0, top - h)
            x2 = min(frame.shape[1], right + w//2)
            y2 = min(frame.shape[0], bottom + h//2)
            detections.append([x1, y1, x2, y2, 0.8])
        return detections

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        xi1, yi1 = max(ax1, bx1), max(ay1, by1)
        xi2, yi2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _track(self, detections):
        prev = self.tracks
        current = {}
        for tid in prev:
            prev[tid]['matched'] = False
        for x1, y1, x2, y2, conf in detections:
            bbox = (x1, y1, x2, y2)
            best_id = None
            best_iou = 0.3
            for tid, tr in prev.items():
                iou = self._iou(bbox, tr['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid
            if best_id is not None:
                current[best_id] = {
                    'bbox': bbox,
                    'last_seen': 0,
                    'person_id': prev[best_id].get('person_id'),
                    'name': prev[best_id].get('name', 'Unknown'),
                    'matched': True,
                    'counted': prev[best_id].get('counted', False),
                    'image_saved': prev[best_id].get('image_saved', False)
                }
                prev[best_id]['matched'] = True
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                current[tid] = {
                    'bbox': bbox,
                    'last_seen': 0,
                    'person_id': None,
                    'name': 'Unknown',
                    'matched': True,
                    'counted': False,
                    'image_saved': False
                }
        for tid, tr in prev.items():
            if not tr.get('matched', False):
                tr['last_seen'] += 1
                if tr['last_seen'] < self.max_age:
                    current[tid] = tr
        removed = set(prev.keys()) - set(current.keys())
        for rid in removed:
            tr = prev.get(rid)
            if tr and tr.get('person_id') is not None and tr.get('counted', False):
                self._on_exit(tr)
        self.tracks = current
        return current

    def _detect_faces_in_bbox(self, frame, bbox):
        if face_recognition is None:
            return []
        x1, y1, x2, y2 = map(int, bbox)
        m = 20
        x1 = max(0, x1 - m)
        y1 = max(0, y1 - m)
        x2 = min(frame.shape[1], x2 + m)
        y2 = min(frame.shape[0], y2 + m)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Note: face_recognition library is CPU-only
        # For production, consider replacing with GPU-based face detection
        locs = face_recognition.face_locations(rgb, model="hog")
        encs = face_recognition.face_encodings(rgb, locs)
        
        faces = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            faces.append({
                'face_location': (top + y1, right + x1, bottom + y1, left + x1),
                'encoding': enc
            })
        return faces

    @staticmethod
    def _face_quality_ok(face_image):
        try:
            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                return False
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            bright = float(np.mean(gray))
            return lap > 50 and 30 < bright < 220
        except Exception:
            return False

    def _identify(self, encoding):
        if not self.known_face_encodings or face_recognition is None:
            return None, float('inf')
        dists = face_recognition.face_distance(self.known_face_encodings, encoding)
        idx = int(np.argmin(dists))
        if dists[idx] < 0.6:
            return self.known_face_names[idx], float(dists[idx])
        return None, float(dists[idx])

    def _on_identity(self, track_id, person_id, name, face_image=None):
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        if not tr.get('counted', False):
            count = self.active_person_counts.get(person_id, 0)
            image_path = None
            if count == 0 and face_image is not None and face_image.size != 0:
                image_path = self._save_person_image(person_id, face_image)
            self._append_presence_event(person_id, name, 'enter', image_path)
            self.active_person_counts[person_id] = count + 1
            tr['counted'] = True
        if face_image is not None and not tr.get('image_saved', False):
            p = self._save_person_image(person_id, face_image)
            if p:
                tr['image_saved'] = True

    def _on_exit(self, track):
        pid = track.get('person_id')
        name = track.get('name', f'Person_{pid}') if pid is not None else 'Unknown'
        if pid is None:
            return
        cnt = self.active_person_counts.get(pid, 0)
        if cnt > 0:
            cnt -= 1
            self.active_person_counts[pid] = cnt
            if cnt == 0:
                self._append_presence_event(pid, name, 'exit')

    def _crop_face(self, frame, loc):
        top, right, bottom, left = loc
        e = 10
        top = max(0, top - e)
        left = max(0, left - e)
        bottom = min(frame.shape[0], bottom + e)
        right = min(frame.shape[1], right + e)
        return frame[top:bottom, left:right]

    def process_frame(self, frame, line_config=None):
        # Use GPU-accelerated YOLO for person detection
        dets = self._detect_people(frame)
        tracks = self._track(dets)
        
        # Draw tripwire line and side labels if enabled
        if line_config and line_config.get('enabled') and all(line_config.get(k) is not None for k in ('x1','y1','x2','y2')):
            x1, y1, x2, y2 = int(line_config['x1']), int(line_config['y1']), int(line_config['x2']), int(line_config['y2'])
            cv2.line(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
            vx, vy = (x2 - x1), (y2 - y1)
            nx, ny = -vy, vx
            norm = (nx**2 + ny**2) ** 0.5 or 1.0
            nx, ny = nx / norm, ny / norm
            offset = 30
            direction = line_config.get('direction', 'AtoBIsIn')
            count_mode = line_config.get('countMode', 'BOTH')
            in_left = (direction == 'AtoBIsIn')
            in_pos = (int(mx + (nx if in_left else -nx) * offset), int(my + (ny if in_left else -ny) * offset))
            out_pos = (int(mx + (-nx if in_left else nx) * offset), int(my + (-ny if in_left else ny) * offset))
            
            def draw_label(img, pos, text, color_bg, color_fg=(255,255,255)):
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.7
                thickness = 2
                (tw, th), bl = cv2.getTextSize(text, font, scale, thickness)
                x, y = pos
                x = max(0, min(img.shape[1] - tw - 16, x - tw // 2))
                y = max(th + 10, min(img.shape[0] - 6, y))
                cv2.rectangle(img, (x - 8, y - th - 10), (x + tw + 8, y + 4), color_bg, -1)
                cv2.putText(img, text, (x, y - 4), font, scale, color_fg, thickness, cv2.LINE_AA)
                
            if count_mode in ('BOTH', 'IN'):
                draw_label(frame, in_pos, 'IN', (32, 114, 187))
            if count_mode in ('BOTH', 'OUT'):
                draw_label(frame, out_pos, 'OUT', (227, 151, 99))
                
        for tid, info in tracks.items():
            bbox = info['bbox']
            faces = self._detect_faces_in_bbox(frame, bbox)
            person_name = info.get('name', 'Unknown')
            color = (0, 0, 255) if person_name == 'Unknown' else (0, 255, 0)
            if faces:
                for face in faces:
                    enc = face['encoding']
                    loc = face['face_location']
                    face_img = self._crop_face(frame, loc)
                    name, dist = self._identify(enc)
                    if name:
                        if self.tracks[tid].get('person_id') is None:
                            pid = None
                            for pid_k, nm in self.person_id_to_name.items():
                                if nm == name:
                                    pid = pid_k
                                    break
                            if pid is None:
                                pid = self.next_person_id
                                self.next_person_id += 1
                                self.person_id_to_name[pid] = name
                            self.tracks[tid]['person_id'] = pid
                            self.tracks[tid]['name'] = name
                            self.face_encoding_cache[pid] = enc
                            self._on_identity(tid, pid, name, face_img)
                        person_name = name
                        color = (0, 255, 0)
                        t, r, b, l = loc
                        cv2.rectangle(frame, (l, t), (r, b), (255, 255, 0), 2)
                    else:
                        buf = self.unknown_faces_buffer.setdefault(tid, [])
                        now = time.time()
                        prev_t = self.creation_cooldown.get(tid, 0)
                        if now - prev_t > 2.0 and self._face_quality_ok(face_img):
                            buf.append({'encoding': enc, 'image': face_img, 'ts': now})
                            if len(buf) >= 3:
                                self._create_new_person_from_track(tid)
                                self.creation_cooldown[tid] = now
                                person_name = self.tracks[tid]['name']
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid} {person_name}"
            if person_name == 'Unknown' and tid in self.unknown_faces_buffer:
                label += f" {len(self.unknown_faces_buffer[tid])}/3"
            if info.get('last_seen', 0) > 0:
                label += " [soon gone]"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Tripwire crossing detection
            if line_config and line_config.get('enabled') and all(line_config.get(k) is not None for k in ('x1','y1','x2','y2')):
                lx1, ly1, lx2, ly2 = int(line_config['x1']), int(line_config['y1']), int(line_config['x2']), int(line_config['y2'])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                side = self._point_side(cx, cy, lx1, ly1, lx2, ly2)
                prev_side = self.track_side.get(tid)
                self.track_side[tid] = side
                if prev_side is not None and prev_side != 0 and side != 0 and np.sign(prev_side) != np.sign(side):
                    crossed_right_to_left = (prev_side < 0 and side > 0)
                    direction = line_config.get('direction', 'AtoBIsIn')
                    event = 'enter' if (direction == 'AtoBIsIn' and crossed_right_to_left) or (direction == 'BtoAIsIn' and not crossed_right_to_left) else 'exit'
                    count_mode = line_config.get('countMode', 'BOTH')
                    if (count_mode == 'IN' and event != 'enter') or (count_mode == 'OUT' and event != 'exit'):
                        continue
                    pid = self.tracks[tid].get('person_id')
                    name = self.tracks[tid].get('name', f'Person_{pid}') if pid is not None else 'Unknown'
                    if pid is not None:
                        snapshot = frame[max(0, cy-60):min(frame.shape[0], cy+60), max(0, cx-60):min(frame.shape[1], cx+60)]
                        img_path = self._save_person_image(pid, snapshot) if snapshot.size != 0 else None
                        self._append_presence_event(pid, name, event, img_path, camera_id=line_config.get('cameraId'))

    @staticmethod
    def _point_side(px, py, x1, y1, x2, y2):
        return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

    def _create_new_person_from_track(self, tid):
        items = self.unknown_faces_buffer.get(tid) or []
        if not items:
            return
        enc = items[0]['encoding']
        imgs = [i.get('image') for i in items if i.get('image') is not None]
        img = imgs[-1] if imgs else None
        pid = self.next_person_id
        name = f"Person_{pid}"
        self.tracks[tid]['person_id'] = pid
        self.tracks[tid]['name'] = name
        self.person_id_to_name[pid] = name
        self.face_encoding_cache[pid] = enc
        self.known_face_encodings.append(enc)
        self.known_face_names.append(name)
        self.next_person_id += 1
        self._save_face_database()
        try:
            del self.unknown_faces_buffer[tid]
        except Exception:
            pass
        self._on_identity(tid, pid, name, img)


# Presence API endpoints remain the same...
@app.get('/api/presence')
def api_presence():
    try:
        if os.path.exists(PRESENCE_LOG_PATH):
            with open(PRESENCE_LOG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return jsonify(data)
    except Exception:
        pass
    return jsonify([])

@app.get('/api/people')
def api_people():
    people = []
    try:
        encs, names = [], []
        if os.path.exists(FACE_DB_PATH):
            with open(FACE_DB_PATH, 'rb') as f:
                data = pickle.load(f)
                encs = data.get('encodings', [])
                names = data.get('names', [])
        for i, nm in enumerate(names):
            person_id = i + 1
            folder = os.path.join(PERSON_IMAGES_DIR, f'person_{person_id}')
            img_count = 0
            if os.path.isdir(folder):
                img_count = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            people.append({'person_id': person_id, 'name': nm, 'images': img_count})
    except Exception:
        pass
    return jsonify(people)

@app.put('/api/people/<int:person_id>')
def api_rename_person(person_id: int):
    payload = request.get_json(force=True, silent=True) or {}
    new_name = str(payload.get('name', '')).strip()
    if not new_name:
        return jsonify({'error': 'name required'}), 400
    try:
        if not os.path.exists(FACE_DB_PATH):
            return jsonify({'error': 'no face database'}), 400
        with open(FACE_DB_PATH, 'rb') as f:
            data = pickle.load(f)
        names = data.get('names', [])
        if person_id < 1 or person_id > len(names):
            return jsonify({'error': 'invalid person_id'}), 400
        names[person_id - 1] = new_name
        data['names'] = names
        with open(FACE_DB_PATH, 'wb') as f:
            pickle.dump(data, f)
    except Exception:
        return jsonify({'error': 'failed to update'}), 500
    try:
        for w in workers.values():
            if getattr(w, 'recognizer', None):
                w.recognizer.reload_face_database_names()
    except Exception:
        pass
    return jsonify({'ok': True, 'person_id': person_id, 'name': new_name})


if __name__ == "__main__":
    # Print GPU status
    print("=" * 50)
    print("GPU ACCELERATION STATUS")
    print("=" * 50)
    if gpu_config.use_gpu:
        print("✅ Running with GPU acceleration")
        print(f"✅ Available GPUs: {gpu_config.available_gpus}")
        for i in range(gpu_config.available_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"     Memory: {gpu_props.total_memory / 1024**3:.1f} GB")
    else:
        print("❌ Running on CPU only")
    print("=" * 50)

    # Ensure config file exists
    config_path = os.path.join(PROJECT_DIR, "config.json")
    if not os.path.exists(config_path):
        default_config = {
            "name": "People Flow",
            "port": 8000,
            "debug": True,
            "cameras": [],
            "showwebcam": False
        }
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
        except OSError:
            pass

    def read_config() -> dict:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "cameras" not in data or not isinstance(data["cameras"], list):
                    data["cameras"] = []
                if "showwebcam" not in data:
                    data["showwebcam"] = False
                return data
        except (OSError, json.JSONDecodeError):
            return {"name": "People Flow", "port": 8000, "debug": True, "cameras": []}

    def write_config(data: dict) -> None:
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    @app.get("/api/cameras")
    def api_get_cameras():
        cfg = read_config()
        return jsonify(cfg.get("cameras", []))

    @app.get("/api/config")
    def api_get_config():
        return jsonify(read_config())

    @app.put("/api/config")
    def api_put_config():
        payload = request.get_json(force=True, silent=True) or {}
        cfg = read_config()
        if "showwebcam" in payload:
            cfg["showwebcam"] = bool(payload["showwebcam"])
            write_config(cfg)
            if cfg["showwebcam"]:
                start_webcam_worker()
            else:
                stop_worker("WEBCAM")
        return jsonify(cfg)

    @app.post("/api/cameras")
    def api_add_camera():
        payload = request.get_json(force=True, silent=True) or {}
        generated_id = f"CAM-{uuid.uuid4().hex[:12].upper()}"
        id_code = str(payload.get("idCode", "")).strip() or generated_id
        cam = Camera(
            idCode=id_code,
            name=str(payload.get("name", "")).strip(),
            location=str(payload.get("location", "")).strip(),
            description=(payload.get("description") or None),
            ipOrHost=str(payload.get("ipOrHost", "")).strip(),
            port=(int(payload["port"]) if str(payload.get("port", "")).strip().isdigit() else None),
            protocol=str(payload.get("protocol", "rtsp")).lower(),
            streamPath=(str(payload.get("streamPath", "")).strip() or None),
            username=(str(payload.get("username")) if payload.get("username") else None),
            passwordEnc=(str(payload.get("password")) if payload.get("password") else None),
            macAddress=(str(payload.get("macAddress")) if payload.get("macAddress") else None),
            mode=str(payload.get("mode", "Recognize")),
            active=bool(payload.get("active", True)),
            lineEnabled=bool(payload.get("lineEnabled", False)),
            lineX1=(int(payload["lineX1"]) if str(payload.get("lineX1", "")).strip().isdigit() else None),
            lineY1=(int(payload["lineY1"]) if str(payload.get("lineY1", "")).strip().isdigit() else None),
            lineX2=(int(payload["lineX2"]) if str(payload.get("lineX2", "")).strip().isdigit() else None),
            lineY2=(int(payload["lineY2"]) if str(payload.get("lineY2", "")).strip().isdigit() else None),
            lineDirection=str(payload.get("lineDirection", "AtoBIsIn")),
            lineCountMode=str(payload.get("lineCountMode", "BOTH")).upper()
        )
        if not cam.idCode:
            return jsonify({"error": "idCode required"}), 400
        if not cam.name:
            return jsonify({"error": "name required"}), 400
        cfg = read_config()
        cams = cfg.get("cameras", [])
        if any(c.get("idCode") == cam.idCode for c in cams):
            return jsonify({"error": "idCode already exists"}), 400
        cams.append(cam.to_dict())
        cfg["cameras"] = cams
        write_config(cfg)
        if cam.active:
            start_worker_for_camera(cam)
        return jsonify(cam.to_dict()), 201

    @app.put("/api/cameras/<id_code>")
    def api_update_camera(id_code: str):
        payload = request.get_json(force=True, silent=True) or {}
        cfg = read_config()
        cams = cfg.get("cameras", [])
        for idx, existing in enumerate(cams):
            if existing.get("idCode") == id_code:
                updated = existing.copy()
                fields = [
                    "name", "location", "description", "ipOrHost", "port",
                    "protocol", "streamPath", "username", "macAddress",
                    "mode", "active", "lineEnabled", "lineX1", "lineY1", "lineX2", "lineY2", "lineDirection", "lineCountMode"
                ]
                for f in fields:
                    if f in payload and payload[f] is not None:
                        updated[f] = payload[f]
                if payload.get("password"):
                    updated["passwordEnc"] = str(payload.get("password"))
                cam = Camera(
                    idCode=updated.get("idCode", id_code),
                    name=updated.get("name", existing.get("name", "")),
                    location=updated.get("location", existing.get("location", "")),
                    description=updated.get("description"),
                    ipOrHost=updated.get("ipOrHost", ""),
                    port=int(updated["port"]) if str(updated.get("port", "")).isdigit() else None,
                    protocol=str(updated.get("protocol", "rtsp")).lower(),
                    streamPath=updated.get("streamPath"),
                    username=updated.get("username"),
                    passwordEnc=updated.get("passwordEnc"),
                    macAddress=updated.get("macAddress"),
                    mode=updated.get("mode", "Recognize"),
                    active=bool(updated.get("active", True)),
                    lineEnabled=bool(updated.get("lineEnabled", False)),
                    lineX1=int(updated["lineX1"]) if str(updated.get("lineX1", "")).isdigit() else None,
                    lineY1=int(updated["lineY1"]) if str(updated.get("lineY1", "")).isdigit() else None,
                    lineX2=int(updated["lineX2"]) if str(updated.get("lineX2", "")).isdigit() else None,
                    lineY2=int(updated["lineY2"]) if str(updated.get("lineY2", "")).isdigit() else None,
                    lineDirection=str(updated.get("lineDirection", "AtoBIsIn")),
                    lineCountMode=str(updated.get("lineCountMode", "BOTH")).upper()
                )
                updated = cam.to_dict()
                cams[idx] = updated
                cfg["cameras"] = cams
                write_config(cfg)
                if updated.get("active", True):
                    start_worker_for_camera(cam)
                else:
                    stop_worker(id_code)
                return jsonify(updated)
        return jsonify({"error": "Camera not found"}), 404

    @app.post("/api/cameras/<id_code>/preview/start")
    def api_start_preview(id_code: str):
        cfg = read_config()
        cams = cfg.get("cameras", [])
        for c in cams:
            if c.get("idCode") == id_code:
                cam = Camera(
                    idCode=c.get("idCode"),
                    name=c.get("name", ""),
                    location=c.get("location", ""),
                    description=c.get("description"),
                    ipOrHost=c.get("ipOrHost", ""),
                    port=int(c["port"]) if str(c.get("port", "")).isdigit() else None,
                    protocol=str(c.get("protocol", "rtsp")).lower(),
                    streamPath=c.get("streamPath"),
                    username=c.get("username"),
                    passwordEnc=c.get("passwordEnc"),
                    macAddress=c.get("macAddress"),
                    mode=c.get("mode", "Recognize"),
                    active=True,
                    lineEnabled=bool(c.get("lineEnabled", False)),
                    lineX1=int(c["lineX1"]) if str(c.get("lineX1", "")).isdigit() else None,
                    lineY1=int(c["lineY1"]) if str(c.get("lineY1", "")).isdigit() else None,
                    lineX2=int(c["lineX2"]) if str(c.get("lineX2", "")).isdigit() else None,
                    lineY2=int(c["lineY2"]) if str(c.get("lineY2", "")).isdigit() else None,
                    lineDirection=str(c.get("lineDirection", "AtoBIsIn")),
                    lineCountMode=str(c.get("lineCountMode", "BOTH")).upper()
                )
                start_worker_for_camera(cam)
                return jsonify({"ok": True})
        return jsonify({"error": "Camera not found"}), 404

    @app.post("/api/cameras/<id_code>/preview/stop")
    def api_stop_preview(id_code: str):
        stop_worker(id_code)
        return jsonify({"ok": True})

    @app.delete("/api/cameras/<id_code>")
    def api_delete_camera(id_code: str):
        cfg = read_config()
        cams = cfg.get("cameras", [])
        new_cams = [c for c in cams if c.get("idCode") != id_code]
        if len(new_cams) == len(cams):
            return jsonify({"error": "Camera not found"}), 404
        cfg["cameras"] = new_cams
        write_config(cfg)
        stop_worker(id_code)
        return jsonify({"ok": True})

    # Start workers for active cameras on server start
    try:
        boot_cfg = read_config()
        for c in boot_cfg.get("cameras", []):
            if c.get("active"):
                cam = Camera(
                    idCode=c.get("idCode"),
                    name=c.get("name", ""),
                    location=c.get("location", ""),
                    description=c.get("description"),
                    ipOrHost=c.get("ipOrHost", ""),
                    port=int(c["port"]) if str(c.get("port", "")).isdigit() else None,
                    protocol=str(c.get("protocol", "rtsp")).lower(),
                    streamPath=c.get("streamPath"),
                    username=c.get("username"),
                    passwordEnc=c.get("passwordEnc"),
                    macAddress=c.get("macAddress"),
                    mode=c.get("mode", "Recognize"),
                    active=True,
                    lineEnabled=bool(c.get("lineEnabled", False)),
                    lineX1=int(c["lineX1"]) if str(c.get("lineX1", "")).isdigit() else None,
                    lineY1=int(c["lineY1"]) if str(c.get("lineY1", "")).isdigit() else None,
                    lineX2=int(c["lineX2"]) if str(c.get("lineX2", "")).isdigit() else None,
                    lineY2=int(c["lineY2"]) if str(c.get("lineY2", "")).isdigit() else None,
                    lineDirection=str(c.get("lineDirection", "AtoBIsIn")),
                    lineCountMode=str(c.get("lineCountMode", "BOTH")).upper()
                )
                start_worker_for_camera(cam)
        if boot_cfg.get("showwebcam"):
            start_webcam_worker()
    except Exception:
        pass

    print(f"🚀 Starting People Flow System on port 8000")
    print(f"📊 GPU Acceleration: {'ENABLED' if gpu_config.use_gpu else 'DISABLED'}")
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True, use_reloader=False)