"""Camera management routes"""

from flask import request, jsonify
from urllib.parse import unquote


def register_camera_routes(app, camera_service, worker_manager):
    """Register camera management routes"""
    
    @app.get("/api/cameras")
    def api_get_cameras():
        """Get all cameras"""
        cameras = camera_service.get_all_cameras()
        response = jsonify(cameras)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    
    @app.post("/api/cameras")
    def api_add_camera():
        """Add a new camera"""
        payload = request.get_json(force=True, silent=True) or {}
        try:
            cam = camera_service.add_camera(payload)
            if cam.active:
                worker_manager.start_worker_for_camera(cam)
            return jsonify(cam.to_dict()), 201
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    
    @app.put("/api/cameras/<id_code>")
    def api_update_camera(id_code: str):
        """Update an existing camera"""
        payload = request.get_json(force=True, silent=True) or {}
        try:
            cam = camera_service.update_camera(id_code, payload)
            
            # Update worker
            existing_worker = worker_manager.get_worker(id_code)
            if existing_worker:
                existing_worker.stop()
            
            if cam.active:
                if existing_worker:
                    existing_worker.cam = cam
                    existing_worker.start(force_restart=True)
                else:
                    worker_manager.start_worker_for_camera(cam)
            else:
                worker_manager.stop_worker(id_code)
            
            return jsonify(cam.to_dict())
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
    
    @app.post("/api/cameras/<id_code>/preview/start")
    def api_start_preview(id_code: str):
        """Start preview for a camera"""
        cam_dict = camera_service.get_camera_by_id(id_code)
        if not cam_dict:
            return jsonify({"error": "Camera not found"}), 404
        
        cam = camera_service.dict_to_camera(cam_dict)
        worker_manager.start_worker_for_camera(cam)
        return jsonify({"ok": True})
    
    @app.post("/api/cameras/<id_code>/preview/stop")
    def api_stop_preview(id_code: str):
        """Stop preview for a camera"""
        worker_manager.stop_worker(id_code)
        return jsonify({"ok": True})
    
    @app.delete("/api/cameras/<id_code>")
    def api_delete_camera(id_code: str):
        """Delete a camera"""
        success = camera_service.delete_camera(id_code)
        if not success:
            return jsonify({"error": "Camera not found"}), 404
        worker_manager.stop_worker(id_code)
        return jsonify({"ok": True})

