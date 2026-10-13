"""Production WSGI entrypoint for Gunicorn."""

from .app import create_app
from .config import load_config

app = create_app(config=load_config())
