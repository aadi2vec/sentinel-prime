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


def test_drain_waits_for_outstanding_children_without_closing_the_pool():
    """A task must not leak worker threads, but the manager outlives one task."""
    import threading
    from sentinelprime.children import ChildSessionManager
    import tempfile

    gate = threading.Event()
    done = []

    def run_child(task, name, session_dir):
        gate.wait(timeout=5)
        done.append(task)
        return f"done:{task}"

    with tempfile.TemporaryDirectory() as tmp:
        mgr = ChildSessionManager(run_child, tmp)
        mgr.spawn_child("a")
        mgr.spawn_child("b")
        assert done == []          # non-blocking admission
        gate.set()
        mgr.drain()
        assert sorted(done) == ["a", "b"]
        # pool still usable afterwards
        handle = mgr.spawn_child("c")
        mgr.drain()
        assert mgr.result(handle.child_id) == "done:c"
        mgr.shutdown()


def test_drain_surfaces_a_child_failure_instead_of_swallowing_it():
    from sentinelprime.children import ChildSessionManager
    import tempfile

    def run_child(task, name, session_dir):
        raise ValueError("child blew up")

    with tempfile.TemporaryDirectory() as tmp:
        mgr = ChildSessionManager(run_child, tmp)
        mgr.spawn_child("a")
        errors = mgr.drain()
        assert len(errors) == 1
        assert isinstance(errors[0][1], ValueError)
        mgr.shutdown()
