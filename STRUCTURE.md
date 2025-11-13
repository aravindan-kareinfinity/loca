# People Flow - Project Structure Documentation

## Overview

The People Flow application has been refactored from a monolithic single-file structure into a clean, modular architecture with clear separation of concerns.

## Directory Structure

```
peopleflow/
├── __init__.py                 # Package initialization
├── app.py                      # Main Flask application entry point
├── example_usage.py            # Example usage script
├── README.md                   # Project documentation
├── STRUCTURE.md                # This file
│
├── models/                     # Data Models Layer
│   ├── __init__.py
│   ├── camera.py              # Camera dataclass with tripwire settings
│   ├── tracker.py             # PersonTracker - IoU + appearance tracking
│   └── counter.py             # PersonCounter - tripwire line crossing logic
│
├── services/                   # Business Logic Layer
│   ├── __init__.py
│   ├── camera_service.py      # Camera CRUD operations, config management
│   ├── worker_service.py      # CameraWorker, WorkerManager, frame processing
│   ├── count_log_service.py   # Count event logging and statistics
│   └── discovery_service.py   # Network device discovery (ARP, MAC vendor lookup)
│
├── routes/                     # API Routes Layer
│   ├── __init__.py            # Route registration function
│   ├── camera_routes.py       # Camera management endpoints
│   ├── stream_routes.py       # MJPEG video stream endpoints
│   ├── api_routes.py          # General API (counts, presence, stats, etc.)
│   └── discovery_routes.py    # Device discovery endpoints
│
└── utils/                     # Utility Functions
    ├── __init__.py
    ├── device_detection.py     # GPU/CPU detection for YOLO
    └── yolo_loader.py         # YOLO model loading and caching
```

## Component Responsibilities

### Models (`models/`)
- **camera.py**: Camera configuration dataclass
  - Connection URL building
  - Tripwire settings (line coordinates, direction, count mode)
  - Serialization to/from dict
  
- **tracker.py**: PersonTracker class
  - IoU-based tracking
  - Appearance feature extraction (HSV histograms)
  - Track management (creation, matching, aging)
  
- **counter.py**: PersonCounter class
  - Point-side line crossing detection
  - IN/OUT counting based on direction
  - Count mode filtering (BOTH/IN/OUT)

### Services (`services/`)
- **camera_service.py**: Camera configuration management
  - Read/write config.json
  - CRUD operations for cameras
  - Camera validation
  
- **worker_service.py**: Video processing and worker management
  - CameraWorker: Frame processing, YOLO detection, tracking, counting
  - WorkerManager: Thread-safe worker registry
  - FFmpeg integration for RTSP streams
  - Frame buffer management
  
- **count_log_service.py**: Count event logging
  - Append count events to JSON log
  - Query events with filtering (date, camera_id, limit)
  - Calculate statistics (totals, remaining)
  - Peak hours analysis
  
- **discovery_service.py**: Network device discovery
  - ARP table parsing
  - MAC vendor lookup via API
  - Camera identification from vendor names

### Routes (`routes/`)
- **camera_routes.py**: Camera management API
  - GET/POST/PUT/DELETE /api/cameras
  - Preview start/stop endpoints
  
- **stream_routes.py**: Video streaming
  - GET /api/stream/<id> - MJPEG stream
  
- **api_routes.py**: General API endpoints
  - /api/config - Configuration
  - /api/counts - People counts
  - /api/count-log - Event log
  - /api/presence - Presence events
  - /api/peak-hours - Peak time analysis
  - /api/camera-performance - Performance metrics
  
- **discovery_routes.py**: Device discovery
  - /api/discover - Discover cameras
  - /api/discover/cameras - All network devices

### Utils (`utils/`)
- **device_detection.py**: Hardware detection
  - GPU/CPU detection for PyTorch/YOLO
  
- **yolo_loader.py**: YOLO model management
  - Model path detection (script vs executable)
  - Model loading and caching
  - Device assignment

## Data Flow

### Camera Stream Processing Flow
1. **CameraWorker.start()** → Starts FFmpeg process
2. **frame_reader_thread** → Reads frames from FFmpeg pipe
3. **FrameBuffer** → Thread-safe frame queue
4. **_process_frame()** → YOLO detection → PersonTracker → PersonCounter
5. **Count events** → CountLogService.append_count_event()
6. **Encoded JPEG** → Stored in worker.latest_jpeg
7. **Stream route** → Serves MJPEG from latest_jpeg

### API Request Flow
1. **Route handler** → Receives HTTP request
2. **Service layer** → Business logic processing
3. **Model/Worker** → Data access or processing
4. **Response** → JSON or MJPEG stream

## Key Design Patterns

1. **Service Layer Pattern**: Business logic separated from routes
2. **Repository Pattern**: Services abstract data access
3. **Worker Pattern**: Background threads for video processing
4. **Singleton Pattern**: YOLO model cached globally
5. **Factory Pattern**: WorkerManager creates/manages workers

## Thread Safety

- **WorkerManager**: Uses threading.Lock for workers dict
- **FrameBuffer**: Uses threading.Condition for frame queue
- **CameraWorker**: Uses threading.Lock for counts and JPEG
- **CountLogService**: File I/O (consider adding locks if concurrent writes)

## Configuration Files

- **config.json**: Camera configurations, app settings
- **count_log.json**: Count event history (last 10,000 events)

## Dependencies

### Python Packages
- Flask: Web framework
- OpenCV (cv2): Image processing
- NumPy: Numerical operations
- Ultralytics YOLO: Person detection
- requests: MAC vendor API calls

### System Dependencies
- FFmpeg: RTSP stream processing
- PyTorch: YOLO model execution (GPU/CPU)

## Migration Notes

The original monolithic code has been split as follows:

- **Camera dataclass** → `models/camera.py`
- **PersonTracker** → `models/tracker.py`
- **PersonCounter** → `models/counter.py`
- **CameraWorker** → `services/worker_service.py`
- **Config management** → `services/camera_service.py`
- **Count logging** → `services/count_log_service.py`
- **Device discovery** → `services/discovery_service.py`
- **All Flask routes** → `routes/*.py`
- **YOLO loading** → `utils/yolo_loader.py`
- **Device detection** → `utils/device_detection.py`
- **Main app** → `app.py`

## Usage

```python
from peopleflow.app import create_app

app = create_app()
app.run(host="0.0.0.0", port=8000)
```

Or use the example:
```bash
python peopleflow/example_usage.py
```

