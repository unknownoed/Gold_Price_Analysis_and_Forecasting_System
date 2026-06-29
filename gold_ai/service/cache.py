import time
import threading


class TTLCache:
    """Thread-safe in-memory TTL cache."""

    def __init__(self, ttl_seconds=300):
        self._store = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._store:
                value, timestamp = self._store[key]
                if time.time() - timestamp < self._ttl:
                    return value
                del self._store[key]
        return None

    def set(self, key, value):
        with self._lock:
            self._store[key] = (value, time.time())

    def invalidate(self, key=None):
        with self._lock:
            if key:
                self._store.pop(key, None)
            else:
                self._store.clear()

    def __len__(self):
        with self._lock:
            return len(self._store)
