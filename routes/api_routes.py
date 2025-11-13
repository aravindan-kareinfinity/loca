"""General API routes"""

from flask import request, jsonify, send_from_directory, render_template
import os
import sys
from datetime import datetime


def register_api_routes(app, camera_service, worker_manager, count_log_service):
    """Register general API routes"""
    
    @app.get("/")
    def serve_root():
        """Serve index.html"""
        return render_template('index.html')
    
    @app.get("/<path:path>")
    def serve_assets(path: str):
        """Serve static assets"""
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "static")
        return send_from_directory(base_dir, path)
    
    @app.get("/api/config")
    def api_get_config():
        """Get application configuration"""
        return jsonify(camera_service.read_config())
    
    @app.put("/api/config")
    def api_put_config():
        """Update application configuration"""
        payload = request.get_json(force=True, silent=True) or {}
        cfg = camera_service.read_config()
        if "showwebcam" in payload:
            cfg["showwebcam"] = bool(payload["showwebcam"])
            camera_service.write_config(cfg)
            # Webcam worker management would go here
        return jsonify(cfg)
    
    @app.get('/api/counts')
    def api_get_counts():
        """Get people counts for all active cameras"""
        counts = {}
        workers = worker_manager.get_all_workers()
        for cam_id, worker in workers.items():
            if worker and worker.cam.active:
                worker_counts = worker.get_counts()
                counts[cam_id] = {
                    'camera_id': cam_id,
                    'camera_name': worker.cam.name,
                    **worker_counts
                }
        return jsonify(counts)
    
    @app.get('/api/counts/<id_code>')
    def api_get_camera_counts(id_code: str):
        """Get people counts for a specific camera"""
        worker = worker_manager.get_worker(id_code)
        if worker is None:
            return jsonify({"error": "Camera not found"}), 404
        return jsonify({
            'camera_id': id_code,
            'camera_name': worker.cam.name,
            **worker.get_counts()
        })
    
    @app.get('/api/count-log')
    def api_get_count_log():
        """Get count log with optional filtering"""
        camera_id = request.args.get('camera_id')
        date_filter = request.args.get('date')
        limit = request.args.get('limit', type=int)
        
        events = count_log_service.get_events(
            camera_id=camera_id,
            date_filter=date_filter,
            limit=limit
        )
        return jsonify(events)
    
    @app.get('/api/count-log/stats')
    def api_get_count_log_stats():
        """Get count statistics"""
        date_filter = request.args.get('date')
        camera_id = request.args.get('camera_id')
        stats = count_log_service.get_stats(date_filter=date_filter, camera_id=camera_id)
        return jsonify(stats)
    
    @app.get('/api/presence')
    def api_get_presence():
        """Get presence events (converted from count log)"""
        date_filter = request.args.get('date')
        limit = request.args.get('limit', type=int)
        
        count_events = count_log_service.get_events(date_filter=date_filter, limit=limit)
        cfg = camera_service.read_config()
        cameras = {c.get("idCode"): c for c in cfg.get("cameras", [])}
        
        # Convert count log events to presence format
        presence_events = []
        for event in count_events:
            camera_id = event.get('camera_id', '')
            camera_name = cameras.get(camera_id, {}).get('name', camera_id)
            event_type = event.get('event', '')
            
            # Convert 'in'/'out' to 'enter'/'exit'
            if event_type == 'in':
                event_type = 'enter'
            elif event_type == 'out':
                event_type = 'exit'
            
            presence_events.append({
                'timestamp': event.get('timestamp', ''),
                'name': f"Camera: {camera_name}",
                'person_id': None,
                'event': event_type,
                'camera_id': camera_id,
                'count_in': event.get('count_in', 0),
                'count_out': event.get('count_out', 0),
                'image_path': None
            })
        
        # Sort by timestamp (newest first)
        presence_events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify(presence_events)
    
    @app.get('/api/peak-hours')
    def api_get_peak_hours():
        """Get peak entry and exit times"""
        return jsonify(count_log_service.get_peak_hours())
    
    @app.get('/api/camera-performance')
    def api_get_camera_performance():
        """Get camera performance metrics"""
        events = count_log_service.read_count_log()
        cfg = camera_service.read_config()
        cameras = cfg.get("cameras", [])
        
        performance = []
        
        for cam in cameras:
            cam_id = cam.get("idCode", "")
            cam_name = cam.get("name", cam_id)
            is_active = cam.get("active", False)
            
            # Count events for this camera
            cam_events = [e for e in events if e.get('camera_id') == cam_id]
            total_events = len(cam_events)
            
            # Check if worker is running
            worker = worker_manager.get_worker(cam_id)
            worker_running = worker is not None and worker.thread and worker.thread.is_alive() if worker else False
            
            # Calculate performance percentage
            if is_active and worker_running:
                if total_events > 0:
                    performance_pct = min(95.0 + (total_events * 0.1), 99.9)
                else:
                    performance_pct = 85.0
            elif is_active:
                performance_pct = 60.0
            else:
                performance_pct = 30.0
            
            performance.append({
                'name': cam_name,
                'performance': round(performance_pct, 1),
                'status': 'good' if performance_pct >= 80 else 'warning' if performance_pct >= 50 else 'poor'
            })
        
        # Sort by performance (highest first)
        performance.sort(key=lambda x: x['performance'], reverse=True)
        
        return jsonify(performance)

