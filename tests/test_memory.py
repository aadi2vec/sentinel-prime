import pytest
from sentinelprime.memory import MemoryItem, JsonMemoryBackend


def _item(id, scope="global", kind="note", text="t"):
    return MemoryItem(id=id, scope=scope, kind=kind, text=text, created_at="2026-01-01", meta={})


def test_write_then_read_returns_items(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a"), _item("b")])
    ids = sorted(i.id for i in be.read())
    assert ids == ["a", "b"]


def test_write_upserts_by_id(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a", text="old")])
    be.write([_item("a", text="new")])
    items = be.read()
    assert len(items) == 1
    assert items[0].text == "new"


def test_read_filters_by_scope(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a", scope="global"), _item("b", scope="session")])
    assert [i.id for i in be.read(scope="session")] == ["b"]


def test_delete_removes_items(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a"), _item("b")])
    be.delete(["a"])
    assert [i.id for i in be.read()] == ["b"]


def test_snapshot_then_mutate_then_rollback_restores(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    be.write([_item("a", text="orig")])
    v = be.snapshot()
    be.write([_item("a", text="changed"), _item("b")])
    assert {i.id for i in be.read()} == {"a", "b"}
    be.rollback(v.number)
    items = be.read()
    assert [i.id for i in items] == ["a"]
    assert items[0].text == "orig"


def test_state_persists_across_reinstantiation(tmp_path):
    path = str(tmp_path / "s.json")
    JsonMemoryBackend(path).write([_item("a")])
    assert [i.id for i in JsonMemoryBackend(path).read()] == ["a"]


def test_rollback_unknown_version_raises(tmp_path):
    be = JsonMemoryBackend(str(tmp_path / "s.json"))
    with pytest.raises(KeyError):
        be.rollback(999)
