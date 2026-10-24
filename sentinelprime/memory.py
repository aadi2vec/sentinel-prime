from __future__ import annotations
import copy
import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass
class MemoryItem:
    """One piece of learned guidance, with the period it was *in force*.

    **Two time axes, and they deliberately live in two different places.** Transaction time
    — when the system came to believe this — is already recorded, append-only, in `AuditLog`;
    every edit to this item has a record there with its own `created_at` and version window.
    Valid time — the period the guidance was actually in force — is what these two fields add,
    and it is the one the ledger could not express.

    Keeping transaction time in the audit log rather than duplicating it here is the reason
    this is bitemporal *as a system* rather than four columns on a row: two copies of the same
    fact drift, and the audit log is the copy invariant 4 makes unfalsifiable.

    Why valid time was needed at all: `_retire()` used to `delete` a lesson that stopped
    clearing its targets, which answers "is this in force now?" by destroying the answer to
    "was it in force then?". CLAUDE.md's invariant 8 already claims the system can *prove the
    guidance was valid at the time*; for the ledger itself that was only reachable through
    `rollback()`, which mutates the ledger to answer a read-only question. Closing an interval
    instead of deleting a row makes the claim true.

    Both fields default to `""` so a ledger written before they existed round-trips unchanged
    — the same additive shape as `AuditRecord.machinery`. An empty `invalid_at` means *still
    in force*, not *unknown*: the common case has to be the cheap one, because every item in
    a healthy ledger is in force.
    """

    id: str
    scope: str
    kind: str
    text: str
    created_at: str
    meta: dict = field(default_factory=dict)
    # Start of the validity interval. Empty -> fall back to `created_at`, so an item seeded
    # before this field existed is in force from when it was recorded rather than nowhere.
    valid_at: str = ""
    # End of the validity interval, exclusive. Empty -> still in force.
    invalid_at: str = ""

    def in_force_at(self, when: str) -> bool:
        """Was this guidance in force at `when`? Half-open: `[valid_at, invalid_at)`.

        Half-open rather than closed so an item and the rewrite that replaced it are never
        both in force for the same instant — a historical read would otherwise surface two
        versions of one lesson and there would be no way to tell which the run actually saw.

        Comparison is lexicographic on the raw strings. That is exact for the ISO-8601 UTC
        that `_now()` produces (fixed width, fixed `+00:00` offset, so byte order *is*
        chronological order) and it is also what lets a hand-seeded fixture use `"2026-01-01"`.
        The limit, stated rather than hidden: timestamps from a *different* producer — a local
        offset, or a different precision — would compare wrong. `_now()` is the only producer
        on the live path.
        """
        start = self.valid_at or self.created_at
        if start and when < start:
            return False
        return not self.invalid_at or when < self.invalid_at


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
