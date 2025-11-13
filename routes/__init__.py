"""People Flow Routes"""

from flask import Blueprint

def register_routes(app, camera_service, worker_manager, count_log_service, discovery_service, project_dir):
    """Register all routes with the Flask app"""
    # Handle both relative and absolute imports
    try:
        from .camera_routes import register_camera_routes
        from .stream_routes import register_stream_routes
        from .api_routes import register_api_routes
        from .discovery_routes import register_discovery_routes
    except ImportError:
        from routes.camera_routes import register_camera_routes
        from routes.stream_routes import register_stream_routes
        from routes.api_routes import register_api_routes
        from routes.discovery_routes import register_discovery_routes
    
    register_camera_routes(app, camera_service, worker_manager)
    register_stream_routes(app, worker_manager)
    register_api_routes(app, camera_service, worker_manager, count_log_service)
    register_discovery_routes(app, camera_service, discovery_service)

