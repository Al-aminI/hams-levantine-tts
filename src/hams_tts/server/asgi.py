"""Module-level ASGI app for `uvicorn hams_tts.server.asgi:app`."""
from .app import create_app

app = create_app()
