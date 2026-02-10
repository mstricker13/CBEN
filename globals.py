from pathlib import Path

"""
This file defines all global variables that are needed by other modules.
"""

# Path to to src directory containing all code
BASE = Path(__file__).parent.resolve()
# Path to "meta_info.json"
META_LOCATION = BASE / "meta_info.json"
