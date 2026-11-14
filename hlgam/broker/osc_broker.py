from __future__ import annotations

from hlgam.models import GestureEvent


class OSCGestureBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, address_prefix: str = "/hlgam") -> None:
        try:
            from pythonosc.udp_client import SimpleUDPClient
        except ImportError as exc:
            raise RuntimeError("OSC broadcasting requires the python-osc optional dependency") from exc
        self.host = host
        self.port = port
        self.address_prefix = address_prefix.rstrip("/")
        self._client = SimpleUDPClient(host, port)

    def publish(self, event: GestureEvent) -> None:
        address = f"{self.address_prefix}/{event.gesture.lower()}"
        self._client.send_message(address, event.to_json())
