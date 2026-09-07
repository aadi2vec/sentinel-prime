from sentinelprime.session import SessionStore


def test_append_and_read_roundtrip(tmp_path):
    store = SessionStore(str(tmp_path / "t.jsonl"))
    store.append({"role": "action", "code": "print(1)"})
    store.append({"role": "observation", "text": "1"})
    events = store.read()
    assert events == [
        {"role": "action", "code": "print(1)"},
        {"role": "observation", "text": "1"},
    ]


def test_read_missing_file_returns_empty(tmp_path):
    store = SessionStore(str(tmp_path / "none.jsonl"))
    assert store.read() == []


def test_iter_yields_in_order(tmp_path):
    store = SessionStore(str(tmp_path / "t.jsonl"))
    for i in range(3):
        store.append({"i": i})
    assert [e["i"] for e in store] == [0, 1, 2]
