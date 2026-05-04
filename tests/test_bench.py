import json
import unittest

from protoagi.evals.bench import EndpointBenchResult, endpoint_results_to_json


class BenchTests(unittest.TestCase):
    def test_endpoint_results_to_json_supports_slots(self) -> None:
        payload = endpoint_results_to_json(
            [EndpointBenchResult(round_index=1, seconds=0.5, chars=10, preview="hello")]
        )
        data = json.loads(payload)
        self.assertEqual(data[0]["round_index"], 1)
        self.assertEqual(data[0]["preview"], "hello")


if __name__ == "__main__":
    unittest.main()
