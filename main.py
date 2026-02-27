import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
API_SERVICE_DIR = REPO_ROOT / "services" / "api"

if str(API_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(API_SERVICE_DIR))

from app.main import app
