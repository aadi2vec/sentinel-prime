from __future__ import annotations
import copy
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass
class MemoryItem:
    id: str
    scope: str
    kind: str
    text: str
    created_at: str
    meta: dict = field(default_factory=dict)


@dataclass
class Version:
    number: int
    parent: int | None
    created_at: str


@runtime_checkable
class MemoryBackend(Protocol):
    def read(self, scope: str | None = None, query: str | None = None) -> list[MemoryItem]: ...
    def write(self, items: list[MemoryItem]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def current_version(self) -> Version: ...
    def snapshot(self) -> Version: ...
    def rollback(self, version: int) -> None: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonMemoryBackend:
    def __init__(self, path: str):
        self.path = path
        self._items: dict[str, MemoryItem] = {}
        self._version = 0
        self._parent: int | None = None
        self._history: dict[int, list[dict]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            raw = json.load(f)
        self._items = {i["id"]: MemoryItem(**i) for i in raw.get("items", [])}
        self._version = raw.get("version", 0)
        self._parent = raw.get("parent_version")
        self._history = {int(k): v for k, v in raw.get("history", {}).items()}

    def _persist(self) -> None:
        raw = {
            "version": self._version,
            "parent_version": self._parent,
            "items": [asdict(i) for i in self._items.values()],
            "history": self._history,
        }
        with open(self.path, "w") as f:
            json.dump(raw, f, indent=2)

    def read(self, scope: str | None = None, query: str | None = None) -> list[MemoryItem]:
        items = list(self._items.values())
        if scope is not None:
            items = [i for i in items if i.scope == scope]
        return items

    def write(self, items: list[MemoryItem]) -> None:
        for it in items:
            self._items[it.id] = it
        self._persist()

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._items.pop(i, None)
        self._persist()

    def current_version(self) -> Version:
        return Version(number=self._version, parent=self._parent, created_at=_now())

    def snapshot(self) -> Version:
        self._parent = self._version
        self._version += 1
        self._history[self._version] = [asdict(i) for i in self._items.values()]
        self._persist()
        return Version(number=self._version, parent=self._parent, created_at=_now())

    def rollback(self, version: int) -> None:
        snap = self._history[version]  # raises KeyError if unknown
        self._items = {i["id"]: MemoryItem(**i) for i in copy.deepcopy(snap)}
        self._persist()
