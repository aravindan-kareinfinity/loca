"""Camera model and dataclass"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
from datetime import datetime
from urllib.parse import quote

ProtocolType = Literal["rtsp", "onvif", "http", "https", "rtmp"]
ModeType = Literal["Recognize", "Identify", "Recognize and Identify"]


@dataclass
class Camera:
    idCode: str
    name: str
    location: str
    description: Optional[str] = None
    ipOrHost: str = ""
    port: Optional[int] = None
    protocol: ProtocolType = "rtsp"
    streamPath: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None  # Store plain password (not encoded)
    macAddress: Optional[str] = None
    mode: ModeType = "Recognize"
    active: bool = True
    createdAt: datetime = field(default_factory=datetime.utcnow)
    updatedAt: datetime = field(default_factory=datetime.utcnow)
    
    # Tripwire settings (screen coordinates in stream frame)
    lineEnabled: bool = False
    lineX1: Optional[int] = None
    lineY1: Optional[int] = None
    lineX2: Optional[int] = None
    lineY2: Optional[int] = None
    lineDirection: str = "AtoBIsIn"
    lineCountMode: str = "BOTH"

    @property
    def streamingUrl(self) -> str:
        return self.connection_url()

    def connection_url(self) -> str:
        """Builds the actual URL for connecting to the camera feed.
        
        Password is stored in plain text and URL-encoded only when building the URL.
        """
        host = (self.ipOrHost or "").strip()
        if not host:
            return ""

        proto = (self.protocol or "rtsp").lower().rstrip(":")
        auth = ""
        if self.username:
            auth = self.username
            if self.password:
                # URL-encode password only when building the URL
                encoded_password = quote(str(self.password), safe='')
                auth = f"{auth}:{encoded_password}"
            auth += "@"

        port = f":{self.port}" if self.port else ""
        path = f"/{(self.streamPath or '').lstrip('/')}" if self.streamPath else ""

        return f"{proto}://{auth}{host}{port}{path}"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["streamingUrl"] = self.streamingUrl
        data["createdAt"] = self.createdAt.isoformat()
        data["updatedAt"] = self.updatedAt.isoformat()
        return data

