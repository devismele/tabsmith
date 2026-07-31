r"""One run directory, one run.

Two runs sharing a run directory write the same checkpoints and the same
pilot-state.json, so one silently corrupts the other's training while both look
healthy in their own logs. This is not hypothetical: during the v3 study a
monitoring check misread a live trainer as dead, a duplicate was launched, and
both were writing the same paths until the duplicate was killed.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ml.full_band.run_pilot import acquire_run_lock, process_is_alive


class ProcessLivenessTests(unittest.TestCase):
    def test_this_process_is_alive(self):
        self.assertTrue(process_is_alive(os.getpid()))

    def test_an_impossible_pid_is_not_alive(self):
        self.assertFalse(process_is_alive(0))
        self.assertFalse(process_is_alive(-1))

    def test_a_very_unlikely_pid_is_reported_dead(self):
        # Not a guarantee on every OS, but a pid this high is not in use here.
        self.assertFalse(process_is_alive(4_294_967_294))


class RunLockTests(unittest.TestCase):
    def test_first_run_acquires_the_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = acquire_run_lock(Path(tmp))
            self.assertTrue(lock.exists())
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["pid"], os.getpid())

    def test_a_live_holder_blocks_a_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A different live pid: the parent process, which is not us.
            other = os.getppid()
            if other in (0, os.getpid()) or not process_is_alive(other):
                self.skipTest("no distinct live pid available to impersonate")
            (Path(tmp) / "run.lock").write_text(
                json.dumps({"pid": other, "started": "now"}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                acquire_run_lock(Path(tmp))

    def test_a_stale_lock_is_taken_over(self):
        """A crashed run must not require manual cleanup."""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "run.lock").write_text(
                json.dumps({"pid": 4_294_967_294, "started": "long ago"}), encoding="utf-8")
            lock = acquire_run_lock(Path(tmp))
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["pid"], os.getpid())

    def test_our_own_lock_is_reentrant(self):
        with tempfile.TemporaryDirectory() as tmp:
            acquire_run_lock(Path(tmp))
            acquire_run_lock(Path(tmp))  # must not raise

    def test_force_takes_over_a_live_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            other = os.getppid()
            if other in (0, os.getpid()) or not process_is_alive(other):
                self.skipTest("no distinct live pid available to impersonate")
            (Path(tmp) / "run.lock").write_text(
                json.dumps({"pid": other, "started": "now"}), encoding="utf-8")
            lock = acquire_run_lock(Path(tmp), force=True)
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["pid"], os.getpid())

    def test_an_unreadable_lock_does_not_wedge_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "run.lock").write_text("{ truncated", encoding="utf-8")
            lock = acquire_run_lock(Path(tmp))
            self.assertEqual(json.loads(lock.read_text(encoding="utf-8"))["pid"], os.getpid())


if __name__ == "__main__":
    unittest.main()
