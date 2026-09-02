import threading
import pytest
from sentinelprime.children import ChildSessionManager, SpawnHandle


def test_spawn_is_non_blocking_then_result_available(tmp_path):
    release = threading.Event()

    def run_child(task, name, session_dir):
        release.wait(timeout=2)
        return f"done:{task}"

    mgr = ChildSessionManager(run_child, str(tmp_path))
    handle = mgr.spawn_child("read the NDA", name="reader")
    assert isinstance(handle, SpawnHandle)
    assert mgr.done(handle.child_id) is False   # returned before child finished
    release.set()
    assert mgr.result(handle.child_id, timeout=2) == "done:read the NDA"
    mgr.shutdown()


def test_depth_guard_rejects_beyond_max(tmp_path):
    mgr = ChildSessionManager(lambda *a: None, str(tmp_path), max_depth=2, depth=2)
    with pytest.raises(RecursionError):
        mgr.spawn_child("too deep")
    mgr.shutdown()


def test_registry_tracks_children(tmp_path):
    mgr = ChildSessionManager(lambda *a: "ok", str(tmp_path))
    h1 = mgr.spawn_child("a")
    h2 = mgr.spawn_child("b")
    ids = {h.child_id for h in mgr.children()}
    assert {h1.child_id, h2.child_id} == ids
    mgr.shutdown()


def test_session_dir_created(tmp_path):
    import os
    mgr = ChildSessionManager(lambda *a: "ok", str(tmp_path))
    handle = mgr.spawn_child("a", name="worker")
    assert os.path.isdir(handle.session_dir)
    assert "worker" in handle.session_dir
    mgr.shutdown()
