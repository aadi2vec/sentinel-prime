from __future__ import annotations
import json
import os
from typing import Iterator


class SessionStore:
    def __init__(self, path: str):
        self.path = path

    def append(self, event: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def read(self) -> list[dict]:
        return list(self)

    def __iter__(self) -> Iterator[dict]:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
