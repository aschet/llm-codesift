# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""Configuration, model lists, and the GPU lock."""
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from codesift import gpulock
from codesift.config import Config, read_model_file, working_depth


class TestConfig(unittest.TestCase):
    def test_scheme_is_added_when_missing(self):
        self.assertEqual(Config(host="localhost:11434").host, "http://localhost:11434")

    def test_existing_scheme_is_preserved(self):
        self.assertEqual(Config(host="https://box:11434/").host, "https://box:11434")

    def test_remote_detection(self):
        for host in ("http://localhost:11434", "http://127.0.0.1:11434"):
            self.assertFalse(Config(host=host).is_remote, host)
        self.assertTrue(Config(host="http://192.168.1.50:11434").is_remote)

    def test_results_directory_is_created_on_demand(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "nested" / "results"
            cfg = Config(results_dir=target)
            self.assertFalse(target.exists())
            path = cfg.path("screen.jsonl")
            self.assertTrue(target.is_dir())
            self.assertEqual(path.name, "screen.jsonl")

    def test_explicit_models_are_used_verbatim(self):
        cfg = Config(models=["a:1", "b:2"])
        self.assertEqual(cfg.resolve_models(), ["a:1", "b:2"])


class TestModelFile(unittest.TestCase):
    def test_comments_and_blank_lines_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "models.txt"
            path.write_text(textwrap.dedent("""\
                # a comment

                qwen3:14b
                granite4:8b   # trailing comment
                """), encoding="utf-8")
            self.assertEqual(read_model_file(path), ["qwen3:14b", "granite4:8b"])

    def test_a_pull_command_names_the_model_it_pulls(self):
        # What `discover --write-models` writes, fed straight back in.
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "models.txt"
            path.write_text(textwrap.dedent("""\
                # suggested by codesift discover
                ollama pull qwen3:14b
                granite4:8b
                """), encoding="utf-8")
            self.assertEqual(read_model_file(path), ["qwen3:14b", "granite4:8b"])


class TestGpuLock(unittest.TestCase):
    def test_lock_path_differs_per_endpoint(self):
        self.assertNotEqual(gpulock.lock_path("http://localhost:11434"),
                            gpulock.lock_path("http://192.168.1.50:11434"))

    def test_second_process_waits_for_the_first(self):
        """Two jobs must not hold the lock at once, or their timings interfere."""
        script = textwrap.dedent("""
            import sys, time
            sys.path.insert(0, %r)
            from codesift import gpulock
            gpulock.acquire("test", endpoint="unit-test-endpoint")
            print("ACQUIRED", time.time(), flush=True)
            time.sleep(float(sys.argv[1]))
        """) % str(Path(__file__).resolve().parents[1] / "src")

        with tempfile.TemporaryDirectory() as td:
            runner = Path(td) / "hold.py"
            runner.write_text(script, encoding="utf-8")
            first = subprocess.Popen([sys.executable, str(runner), "3"],
                                     stdout=subprocess.PIPE, text=True)
            try:
                self.assertIn("ACQUIRED", first.stdout.readline())
                started = time.time()
                second = subprocess.run([sys.executable, str(runner), "0"],
                                        capture_output=True, text=True, timeout=30)
                waited = time.time() - started
            finally:
                first.kill()
                first.wait(timeout=10)
                first.stdout.close()
            self.assertIn("ACQUIRED", second.stdout)
            self.assertGreater(waited, 1.0,
                               "second process did not wait for the lock to be released")


class TestWorkingDepth(unittest.TestCase):
    """The depth measurements follow the window the user chose.

    A depth fixed at 48,000 tokens was only coherent against the 65,536 it was
    picked for. Anyone running a smaller window -- which is the usual reason to
    change it, since the KV cache is what pushes a model off the GPU -- got a
    prompt that could not fit, so every model read as truncated and the context
    gate rejected the whole field.
    """

    WINDOWS = (65536, 32768, 16384, 131072)

    def test_the_prompt_always_fits_the_window(self):
        for ctx in self.WINDOWS:
            with self.subTest(ctx=ctx):
                self.assertLess(working_depth(ctx), ctx)

    def test_room_is_left_for_the_reply(self):
        # Filling the window leaves nothing to answer with, and Ollama drops the
        # overflow silently rather than refusing the request.
        for ctx in self.WINDOWS:
            with self.subTest(ctx=ctx):
                self.assertGreaterEqual(ctx - working_depth(ctx), ctx // 8)

    def test_the_depth_is_most_of_the_window(self):
        # Too shallow and the measurement stops resembling a coding session.
        for ctx in self.WINDOWS:
            with self.subTest(ctx=ctx):
                self.assertGreater(working_depth(ctx), ctx // 2)

    def test_a_32k_window_measures_at_24k(self):
        self.assertEqual(working_depth(32768), 24576)


if __name__ == "__main__":
    unittest.main()
