"""Count log management service"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional


class CountLogService:
    """Service for managing count log events"""
    
    def __init__(self, log_path: str):
        self.log_path = log_path
    
    def read_count_log(self) -> List[dict]:
        """Read count log from JSON file"""
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
        except Exception:
            pass
        return []
    
    def write_count_log(self, events: List[dict]) -> None:
        """Write count log to JSON file"""
        try:
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(events, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass
    
    def append_count_event(self, camera_id: str, event_type: str, count_in: int, count_out: int) -> None:
        """Append a count event to the log"""
        try:
            events = self.read_count_log()
            event = {
                "camera_id": camera_id,
                "event": event_type,  # "in" or "out"
                "count_in": count_in,
                "count_out": count_out,
                "timestamp": datetime.now().isoformat()
            }
            events.append(event)
            # Keep only last 10000 events to prevent file from growing too large
            if len(events) > 10000:
                events = events[-10000:]
            self.write_count_log(events)
        except Exception:
            pass
    
    def get_events(self, camera_id: Optional[str] = None, date_filter: Optional[str] = None, limit: Optional[int] = None) -> List[dict]:
        """Get events with optional filtering"""
        events = self.read_count_log()
        
        # Filter by camera_id
        if camera_id:
            events = [e for e in events if e.get('camera_id') == camera_id]
        
        # Filter by date
        if date_filter:
            filtered_events = []
            for e in events:
                timestamp = e.get('timestamp', '')
                if timestamp:
                    try:
                        event_date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                        filter_date = datetime.fromisoformat(date_filter).date()
                        if event_date == filter_date:
                            filtered_events.append(e)
                    except Exception:
                        continue
            events = filtered_events
        
        # Limit results
        if limit and limit > 0:
            events = events[-limit:]
        
        return events
    
    def get_stats(self, date_filter: Optional[str] = None, camera_id: Optional[str] = None) -> Dict:
        """Get count statistics (total IN, OUT, remaining)"""
        events = self.get_events(camera_id=camera_id, date_filter=date_filter)
        
        # Calculate totals from the last event of each camera (they contain cumulative counts)
        total_in = 0
        total_out = 0
        
        # Get the latest count for each camera from the filtered events
        camera_last_event = {}
        for event in events:
            cam_id = event.get('camera_id', '')
            if cam_id:
                timestamp = event.get('timestamp', '')
                if timestamp:
                    try:
                        event_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        if cam_id not in camera_last_event:
                            camera_last_event[cam_id] = {'event': event, 'time': event_time}
                        else:
                            if event_time > camera_last_event[cam_id]['time']:
                                camera_last_event[cam_id] = {'event': event, 'time': event_time}
                    except Exception:
                        continue
        
        # Sum up counts from the last event of each camera
        for cam_id, last_event_data in camera_last_event.items():
            event = last_event_data['event']
            total_in += event.get('count_in', 0)
            total_out += event.get('count_out', 0)
        
        # If no events with count_in/count_out found, calculate from individual events
        if total_in == 0 and total_out == 0 and events:
            for event in events:
                if event.get('event') == 'in':
                    total_in += 1
                elif event.get('event') == 'out':
                    total_out += 1
        
        remaining = max(0, total_in - total_out)
        
        return {
            'total_in': total_in,
            'total_out': total_out,
            'remaining': remaining,
            'date': date_filter if date_filter else None
        }
    
    def get_peak_hours(self) -> Dict[str, str]:
        """Get peak entry and exit times from count log data"""
        events = self.read_count_log()
        if not events:
            return {
                'peak_entry_time': '--',
                'peak_exit_time': '--'
            }
        
        # Group events by hour
        entry_hours = {}  # {hour: count}
        exit_hours = {}   # {hour: count}
        
        for event in events:
            try:
                timestamp = event.get('timestamp', '')
                if not timestamp:
                    continue
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                hour = dt.hour
                event_type = event.get('event', '')
                
                if event_type == 'in':
                    entry_hours[hour] = entry_hours.get(hour, 0) + 1
                elif event_type == 'out':
                    exit_hours[hour] = exit_hours.get(hour, 0) + 1
            except Exception:
                continue
        
        # Find peak hours
        peak_entry_hour = max(entry_hours.items(), key=lambda x: x[1])[0] if entry_hours else None
        peak_exit_hour = max(exit_hours.items(), key=lambda x: x[1])[0] if exit_hours else None
        
        # Format times
        def format_time(hour):
            if hour is None:
                return '--'
            period = 'AM' if hour < 12 else 'PM'
            display_hour = hour if hour <= 12 else hour - 12
            if display_hour == 0:
                display_hour = 12
            return f"{display_hour}:00 {period}"
        
        return {
            'peak_entry_time': format_time(peak_entry_hour),
            'peak_exit_time': format_time(peak_exit_hour)
        }

