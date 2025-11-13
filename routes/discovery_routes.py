"""Device discovery routes"""

from flask import jsonify


def register_discovery_routes(app, camera_service, discovery_service):
    """Register discovery routes"""
    
    @app.get('/api/discover')
    def discover_cameras_api():
        """Discover devices on the LAN and flag likely cameras"""
        existing_cameras = camera_service.get_all_cameras()
        discovered = discovery_service.discover_cameras(existing_cameras)
        return jsonify(discovered)
    
    @app.get('/api/discover/cameras')
    def discover_only_cameras_api():
        """Discover and return all devices on the network"""
        devices = discovery_service.find_lan_devices()
        all_devices = []
        for ip_address, mac, vendor in devices:
            all_devices.append({
                'ip_address': ip_address,
                'mac_address': mac,
                'vendor': vendor,
            })
        return jsonify(all_devices)

