from pathlib import Path

# Project root = the folder that contains src/
ROOT = Path(__file__).resolve().parent.parent

# Every agent run gets its own folder inside runs/
RUNS_DIR = ROOT / "runs"

# Sandbox settings
SANDBOX_IMAGE = "ads-sandbox"        # name of the Docker image we build below
SANDBOX_TIMEOUT_SEC = 90             # kill code that runs longer than this
SANDBOX_MEMORY = "1g"                # RAM limit for the container
SANDBOX_CPUS = "1"                   # CPU limit
MAX_OUTPUT_CHARS = 4000              # cap on output we keep (LLMs have limited context)
