import sys
import warnings
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")
