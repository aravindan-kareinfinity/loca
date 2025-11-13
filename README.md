# People Flow Application

A Flask-based people counting and tracking system with camera management, real-time streaming, and analytics.

## Project Structure

```
peopleflow/
├── __init__.py              # Package initialization
├── app.py                   # Main Flask application entry point
├── models/                  # Data models
│   ├── __init__.py
│   ├── camera.py           # Camera dataclass model
│   ├── tracker.py          # PersonTracker class
│   └── counter.py          # PersonCounter class
├── services/                # Business logic services
│   ├── __init__.py
│   ├── camera_service.py    # Camera configuration management
│   ├── worker_service.py    # Camera worker and frame processing
│   ├── count_log_service.py # Count log management
│   └── discovery_service.py # Network device discovery
├── routes/                  # Flask route handlers
│   ├── __init__.py
│   ├── camera_routes.py     # Camera CRUD endpoints
│   ├── stream_routes.py     # Video stream endpoints
│   ├── api_routes.py       # General API endpoints
│   └── discovery_routes.py  # Device discovery endpoints
└── utils/                  # Utility functions
    ├── __init__.py
    ├── device_detection.py  # GPU/CPU detection
    └── yolo_loader.py      # YOLO model loading
```

## Components

### Models
- **Camera**: Camera configuration dataclass with tripwire settings
- **PersonTracker**: Tracks people across frames using IoU + appearance features
- **PersonCounter**: Counts people crossing tripwire lines

### Services
- **CameraService**: Manages camera configurations (CRUD operations)
- **WorkerService**: Manages camera workers and frame processing
- **CountLogService**: Manages count event logging and statistics
- **DiscoveryService**: Discovers devices on the network

### Routes
- **camera_routes**: `/api/cameras` endpoints for camera management
- **stream_routes**: `/api/stream/<id>` for MJPEG video streams
- **api_routes**: General API endpoints (counts, presence, stats, etc.)
- **discovery_routes**: `/api/discover` for network device discovery

## Usage

### Running the Application

```python
from peopleflow.app import create_app

app = create_app()
app.run(host="0.0.0.0", port=8000, debug=True)
```

Or directly:

```bash
python -m peopleflow.app
```

### Configuration

The application uses:
- `config.json`: Camera configurations and settings
- `count_log.json`: Count event log

Both files are stored in the project directory.

## API Endpoints

### Camera Management
- `GET /api/cameras` - Get all cameras
- `POST /api/cameras` - Add new camera
- `PUT /api/cameras/<id>` - Update camera
- `DELETE /api/cameras/<id>` - Delete camera
- `POST /api/cameras/<id>/preview/start` - Start camera preview
- `POST /api/cameras/<id>/preview/stop` - Stop camera preview

### Streaming
- `GET /api/stream/<id>` - MJPEG stream for camera

### Analytics
- `GET /api/counts` - Get counts for all cameras
- `GET /api/counts/<id>` - Get counts for specific camera
- `GET /api/count-log` - Get count log events
- `GET /api/count-log/stats` - Get count statistics
- `GET /api/presence` - Get presence events
- `GET /api/peak-hours` - Get peak entry/exit times
- `GET /api/camera-performance` - Get camera performance metrics

### Discovery
- `GET /api/discover` - Discover cameras on network
- `GET /api/discover/cameras` - Get all network devices

## Dependencies

- Flask
- OpenCV (cv2)
- NumPy
- Ultralytics YOLO
- FFmpeg (system dependency)

## Notes

- The application supports both script and PyInstaller executable modes
- YOLO model path is automatically detected
- GPU/CPU device is automatically detected for YOLO
- Camera workers process frames asynchronously
- Count events are logged to JSON file

