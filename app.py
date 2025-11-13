"""People Flow Flask Application - Main Entry Point"""

import os
import sys
from flask import Flask, send_from_directory

# Handle PyInstaller bundle path
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    BASE_DIR = sys._MEIPASS
    PROJECT_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    # Running as script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = BASE_DIR
    # Add project directory to Python path for imports when running as script
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

# Paths
COUNT_LOG_PATH = os.path.join(PROJECT_DIR, "count_log.json")
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")

# Ensure PROJECT_DIR exists
os.makedirs(PROJECT_DIR, exist_ok=True)

# Initialize Flask app
if getattr(sys, 'frozen', False):
    static_folder = BASE_DIR
    template_folder = BASE_DIR
else:
    static_folder = os.path.join(PROJECT_DIR, "static")
    template_folder = os.path.join(PROJECT_DIR, "templates")

app = Flask(__name__, static_folder=static_folder, static_url_path="/", template_folder=template_folder)

# Initialize services
# Handle both relative imports (when run as module) and absolute imports (when run as script)
try:
    from .services.camera_service import CameraService
    from .services.worker_service import WorkerManager
    from .services.count_log_service import CountLogService
    from .services.discovery_service import DiscoveryService
    from .routes import register_routes
except ImportError:
    # Running as script, use absolute imports
    from services.camera_service import CameraService
    from services.worker_service import WorkerManager
    from services.count_log_service import CountLogService
    from services.discovery_service import DiscoveryService
    from routes import register_routes

camera_service = CameraService(CONFIG_PATH)
count_log_service = CountLogService(COUNT_LOG_PATH)
discovery_service = DiscoveryService()
worker_manager = WorkerManager(count_log_service=count_log_service)

# Register routes
register_routes(app, camera_service, worker_manager, count_log_service, discovery_service, PROJECT_DIR)

# Start workers for active cameras on server start
def start_active_cameras():
    """Start workers for all active cameras"""
    try:
        boot_cfg = camera_service.read_config()
        for c in boot_cfg.get("cameras", []):
            if c.get("active"):
                cam = camera_service.dict_to_camera(c)
                worker_manager.start_worker_for_camera(cam)
        
        # Start webcam worker if enabled
        if boot_cfg.get("showwebcam"):
            # Webcam worker would be started here
            pass
    except Exception:
        pass

# Initialize on import
start_active_cameras()


def create_app():
    """Factory function to create Flask app"""
    return app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True, use_reloader=False)

