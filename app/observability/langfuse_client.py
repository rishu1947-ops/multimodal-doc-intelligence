# app/observability/langfuse_client.py
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langfuse import Langfuse

# --- Step 1: Find the project root directory ---
# This gets the directory where 'app' is located.
# If the 'app' folder is in your project root, this will find it.
def find_project_root() -> Path:
    # Start from the directory of the current file
    current_dir = Path(__file__).resolve().parent
    # Traverse up until we find a directory containing a .env file
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / '.env').exists():
            return parent
    # Fallback to the directory containing 'app' if no .env is found
    # This assumes a standard structure: project_root/app/subdirs/...
    for parent in current_dir.parents:
        if (parent / 'app').exists():
            return parent
    # Default to the current working directory as a last resort
    return Path.cwd()

# --- Step 2: Load the .env file using the found root path ---
PROJECT_ROOT = find_project_root()
env_path = PROJECT_ROOT / '.env'

if env_path.exists():
    print(f"Loading .env from {env_path}")  # Helpful for debugging
    load_dotenv(dotenv_path=env_path, override=True)
else:
    print(f"⚠️ .env file not found at {env_path}")

# --- Step 3: Initialize the Langfuse client ---
public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST")

if not public_key or not secret_key:
    raise ValueError(
        "Langfuse credentials not found. "
        "Please ensure LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
        "are set in your .env file in the project root: "
        f"{PROJECT_ROOT / '.env'}"
    )

langfuse = Langfuse(
    public_key=public_key,
    secret_key=secret_key,
    host=host
)

def flush():
    """Flush all pending events to Langfuse."""
    langfuse.flush()