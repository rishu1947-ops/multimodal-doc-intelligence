# app/observability/langfuse_client.py
import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
secret_key = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST")

# Only initialise if both keys are present
if public_key and secret_key:
    langfuse = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host
    )
    def flush():
        langfuse.flush()
else:
    # Create a dummy object that does nothing
    class DummyLangfuse:
        def trace(self, *args, **kwargs):
            return DummyTrace()
    class DummyTrace:
        def span(self, *args, **kwargs):
            return self
        def update(self, *args, **kwargs):
            return self
        def end(self):
            return self
        def score(self, *args, **kwargs):
            return self
    langfuse = DummyLangfuse()
    flush = lambda: None