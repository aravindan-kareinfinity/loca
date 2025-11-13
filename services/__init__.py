"""People Flow Services"""

# Handle both relative and absolute imports
try:
    from .camera_service import CameraService
    from .worker_service import WorkerManager
    from .count_log_service import CountLogService
    from .discovery_service import DiscoveryService
except ImportError:
    from services.camera_service import CameraService
    from services.worker_service import WorkerManager
    from services.count_log_service import CountLogService
    from services.discovery_service import DiscoveryService

__all__ = ['CameraService', 'WorkerManager', 'CountLogService', 'DiscoveryService']

