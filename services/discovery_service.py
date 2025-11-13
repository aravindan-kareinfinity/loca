"""Device discovery service"""

import os
import re
from typing import List, Tuple, Optional

try:
    import requests
except Exception:
    requests = None

# Known camera vendors to help classify devices discovered on the LAN
KNOWN_CAMERA_VENDORS = [
    'axis', 'hikvision', 'dahua', 'uniview', 'cp plus',
    'bosch', 'hanwha', 'sony', 'avtech', 'arecont', 'vivotek', 'mobotix',
]


class DiscoveryService:
    """Service for discovering devices on the network"""
    
    @staticmethod
    def get_mac_vendor(mac_address: str) -> str:
        """Lookup vendor name for a MAC address using macvendors API.
        
        Returns 'Unknown' on error.
        """
        if requests is None:
            return "Unknown"
        try:
            url = f"https://api.macvendors.com/{mac_address}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200 and response.text:
                return response.text.strip()
        except Exception:
            pass
        return "Unknown"
    
    @staticmethod
    def find_lan_devices() -> List[Tuple[str, str, str]]:
        """Discover devices using ARP table. Returns list of (ip, mac, vendor).
        
        Flow:
        1. Executes "arp -a" command to get ARP table
        2. Extracts IP and MAC addresses from ARP output
        3. For each device:
           - Normalizes MAC address format
           - Looks up MAC vendor via API
        4. Returns list of (ip_address, mac_address, vendor)
        """
        try:
            output = os.popen("arp -a").read()
        except Exception:
            output = ""
        
        devices = re.findall(r"(\d+\.\d+\.\d+\.\d+)\s+([0-9a-f\-:]{17})", output, re.I)
        results = []
        
        for ip_address, mac in devices:
            normalized_mac = mac.replace('-', ':').lower()
            vendor = DiscoveryService.get_mac_vendor(normalized_mac)
            results.append((ip_address, normalized_mac, vendor))
        
        return results
    
    @staticmethod
    def discover_cameras(existing_cameras: List[dict]) -> List[dict]:
        """Discover devices on the LAN and flag likely cameras.
        
        Args:
            existing_cameras: List of existing camera configurations
            
        Returns:
            List of discovered devices with camera flags
        """
        devices = DiscoveryService.find_lan_devices()
        discovered = []
        
        for ip_address, mac, vendor in devices:
            vendor_lower = vendor.lower()
            is_camera = any(v in vendor_lower for v in KNOWN_CAMERA_VENDORS)
            
            status = "can_add"
            existing_camera_id = None
            
            if is_camera:
                for existing in existing_cameras:
                    existing_ip = (existing.get("ipOrHost") or "").strip().lower()
                    existing_mac = (existing.get("macAddress") or "").strip().lower().replace('-', ':')
                    normalized_mac = mac.replace('-', ':').lower()
                    
                    if (existing_ip and existing_ip == ip_address.lower()) or \
                       (existing_mac and existing_mac == normalized_mac):
                        status = "already_added"
                        existing_camera_id = existing.get("idCode")
                        break
            
            discovered.append({
                'ip_address': ip_address,
                'mac_address': mac,
                'vendor': vendor,
                'is_camera': is_camera,
                'status': status,
                'existing_camera_id': existing_camera_id,
            })
        
        return discovered

