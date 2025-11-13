"""Stream routes for camera feeds"""

from flask import Response, jsonify
import time


def register_stream_routes(app, worker_manager):
    """Register stream routes"""
    
    @app.get('/api/stream/<id_code>')
    def stream_camera(id_code: str):
        """Stream camera feed as MJPEG"""
        worker = worker_manager.get_worker(id_code)
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

