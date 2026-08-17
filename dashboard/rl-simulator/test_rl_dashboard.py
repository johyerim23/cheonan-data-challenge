from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
BUILD_PATH = HERE / "build_rl_dashboard.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_rl_dashboard", BUILD_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("builder import failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RlDashboardBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = load_builder()
        cls.output = cls.builder.build()
        cls.html = cls.output.read_text(encoding="utf-8")

    def test_standalone_rl_dashboard_is_generated(self):
        self.assertGreater(self.output.stat().st_size, 500_000)
        self.assertIn("천안 응급의료 — DQN RL 배치 시뮬레이터", self.html)
        self.assertIn("DQN 정책 모델", self.html)
        self.assertIn("실제 Q-network 단계별 추론", self.html)
        self.assertIn("RL 검증 대시보드", self.html)

    def test_gnn_model_payload_and_inference_are_removed(self):
        self.assertNotIn("const GW=", self.html)
        self.assertNotIn("GNN 대체모델 — 브라우저 추론", self.html)
        self.assertNotIn("function gnnPredict()", self.html)

    def test_rl_policy_and_metrics_are_embedded(self):
        for name in ("수신면", "동면", "서울대정병원"):
            self.assertIn(name, self.html)
        for value in ("19.298895274221845", "17.542134545961957", "1.7567607282598878"):
            self.assertIn(value, self.html)
        self.assertIn('"state_dim":396', self.html)
        self.assertIn('"action_dim":54', self.html)

    def test_real_q_value_trace_matches_dqn_actions(self):
        trace_data = json.loads((HERE / "dqn_trace.json").read_text(encoding="utf-8"))
        expected = {
            42: [51, 24, 25],
            123: [25, 51, 24],
            2026: [25, 24, 51],
        }
        self.assertEqual(len(trace_data["traces"]), 3)
        for trace in trace_data["traces"]:
            selected = [step["selected_action_index"] for step in trace["steps"]]
            self.assertEqual(selected, expected[trace["seed"]])
            for step in trace["steps"]:
                self.assertEqual(step["ranked_actions"][0]["action_index"], step["selected_action_index"])
                self.assertTrue(step["ranked_actions"][0]["selected"])

    def test_build_is_deterministic(self):
        first = hashlib.sha256(self.output.read_bytes()).hexdigest()
        self.builder.build()
        second = hashlib.sha256(self.output.read_bytes()).hexdigest()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
