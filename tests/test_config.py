"""Configuration, model lists, and the GPU lock."""
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from codesift import gpulock
from codesift.config import Config, parse_kv, read_model_file


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


class TestParseKv(unittest.TestCase):
    def test_basic_key_value(self):
        self.assertEqual(parse_kv("a=b"), {"a": "b"})

    def test_multiple_pairs(self):
        result = parse_kv("x=1\ny=2")
        self.assertEqual(result, {"x": "1", "y": "2"})

    def test_whitespace_is_stripped(self):
        self.assertEqual(parse_kv("  a  =  b  "), {"a": "b"})

    def test_value_can_contain_equals(self):
        self.assertEqual(parse_kv("a=b=c"), {"a": "b=c"})

    def test_blank_lines_ignored(self):
        result = parse_kv("\n\nc=d\n\n")
        self.assertEqual(result, {"c": "d"})

    def test_comments_ignored(self):
        result = parse_kv("# hello\nx=y")
        self.assertEqual(result, {"x": "y"})

    def test_commented_blank_combo(self):
        text = "# comment\n\n  \nz=1"
        self.assertEqual(parse_kv(text), {"z": "1"})

    def test_no_equals_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            parse_kv("nospaces")
        self.assertIn("nospaces", str(ctx.exception))

    def test_empty_text_returns_empty_dict(self):
        self.assertEqual(parse_kv(""), {})


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


if __name__ == "__main__":
    unittest.main()
