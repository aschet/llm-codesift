"""HTTP client behaviour, exercised without a server."""
import copy
import unittest
import urllib.error

from codesift.ollama import Ollama


class RecordingClient(Ollama):
    """Captures requests and replays canned responses instead of using the network."""

    def __init__(self, responses=None, fail_on_think=False):
        super().__init__("http://testhost:11434")
        self.requests = []
        self.responses = list(responses or [])
        self.fail_on_think = fail_on_think

    def _post(self, path, payload, timeout=None):
        # Copy: the retry path mutates the caller's payload, which would
        # otherwise rewrite the request already recorded.
        self.requests.append((path, copy.deepcopy(payload)))
        if self.fail_on_think and "think" in payload:
            raise urllib.error.HTTPError(path, 400, "Bad Request", {}, None)
        return self.responses.pop(0) if self.responses else {}

    def _get(self, path, timeout=30):
        self.requests.append((path, None))
        return self.responses.pop(0) if self.responses else {}


class TestChatPayload(unittest.TestCase):
    def test_determinism_settings_are_sent(self):
        client = RecordingClient([{}])
        client.chat("m", "hello", ctx=4096, num_predict=64)
        _, payload = client.requests[0]
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(payload["options"]["seed"], 1)
        self.assertEqual(payload["options"]["num_ctx"], 4096)
        self.assertEqual(payload["options"]["num_predict"], 64)
        self.assertFalse(payload["stream"])

    def test_tools_are_only_sent_when_provided(self):
        client = RecordingClient([{}, {}])
        client.chat("m", "hi", ctx=1, num_predict=1)
        self.assertNotIn("tools", client.requests[0][1])
        client.chat("m", "hi", ctx=1, num_predict=1, tools=[{"type": "function"}])
        self.assertIn("tools", client.requests[1][1])

    def test_think_field_is_retried_without_it(self):
        """Some models reject `think`; the request must succeed anyway."""
        client = RecordingClient([{"message": {"content": "ok"}}], fail_on_think=True)
        result = client.chat("m", "hi", ctx=1, num_predict=1, think=False)
        self.assertEqual(len(client.requests), 2, "should retry once")
        self.assertIn("think", client.requests[0][1])
        self.assertNotIn("think", client.requests[1][1])
        self.assertEqual(result["message"]["content"], "ok")

    def test_wall_time_is_attached(self):
        client = RecordingClient([{}])
        self.assertIn("_wall", client.chat("m", "hi", ctx=1, num_predict=1))


class TestPlacement(unittest.TestCase):
    def test_vram_split_is_computed(self):
        client = RecordingClient([{"models": [
            {"name": "target", "size": 20_000_000_000, "size_vram": 5_000_000_000}]}])
        placement = client.placement("target")
        self.assertEqual(placement["pct_gpu"], 25.0)
        self.assertEqual(placement["total_gb"], 20.0)

    def test_model_not_loaded(self):
        client = RecordingClient([{"models": []}])
        self.assertEqual(client.placement("absent"), {})

    def test_zero_size_does_not_divide_by_zero(self):
        client = RecordingClient([{"models": [
            {"name": "odd", "size": 0, "size_vram": 0}]}])
        self.assertIsNone(client.placement("odd")["pct_gpu"])


class TestUnload(unittest.TestCase):
    def test_unload_requests_zero_keep_alive(self):
        client = RecordingClient([{}])
        client.unload("m")
        self.assertEqual(client.requests[0][1]["keep_alive"], 0)

    def test_unload_all_stops_when_clear(self):
        client = RecordingClient([{"models": []}])
        self.assertTrue(client.unload_all(deadline=5))

    def test_unload_all_gives_up_after_the_deadline(self):
        class Stuck(RecordingClient):
            def _get(self, path, timeout=30):
                return {"models": [{"name": "stuck"}]}
            def _post(self, path, payload, timeout=None):
                return {}
        self.assertFalse(Stuck().unload_all(deadline=0.1))

    def test_errors_during_unload_are_swallowed(self):
        class Broken(RecordingClient):
            def _post(self, path, payload, timeout=None):
                raise OSError("connection reset")
        Broken().unload("m")      # must not raise


if __name__ == "__main__":
    unittest.main()
