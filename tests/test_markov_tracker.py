import unittest

import torch

from surgmark.data.jsonl_dataset import build_label_space
from surgmark.model.observer import SurgMarkObserver
from surgmark.streaming.markov_tracker import MarkovStateTracker


def record(video_id, start, end, atom, step, phase):
    return {
        "video_id": video_id,
        "time": {"start_sec": start, "end_sec": end},
        "state": {
            "atom": atom,
            "step": step,
            "cluster": f"{phase}-C01",
            "phase": phase,
            "node_name": atom,
        },
    }


class LabelSpacePriorTest(unittest.TestCase):
    def test_training_sequences_generate_normalized_priors(self):
        records = [
            record("v1", 0, 20, "A", "P1-C01-S01", "P1"),
            record("v1", 30, 50, "B", "P2-C01-S01", "P2"),
            record("v2", 0, 20, "A", "P1-C01-S01", "P1"),
            record("v2", 30, 50, "A", "P1-C01-S01", "P1"),
            record("v3", 0, 20, "A", "P1-C01-S01", "P1"),
            record("v3", 30, 50, "A", "P1-C01-S01", "P1"),
        ]
        label_space = build_label_space(records, epsilon=1e-6, duration_bin_size=30.0)
        priors = label_space["priors"]

        self.assertAlmostEqual(sum(priors["empirical_transitions"]["A"].values()), 1.0)
        self.assertAlmostEqual(sum(priors["procedural_transitions"]["A"].values()), 1.0)
        self.assertGreater(
            priors["empirical_transitions"]["A"]["A"],
            priors["empirical_transitions"]["A"]["B"],
        )
        self.assertIn("A", priors["duration_distributions"])


class MarkovBeliefTest(unittest.TestCase):
    def setUp(self):
        self.label_space = {
            "levels": {
                "phase": ["P1", "P2"],
                "cluster": ["P1-C01", "P2-C01"],
                "step": ["P1-C01-S01", "P2-C01-S01"],
                "atom": ["A", "B"],
            },
            "node_names": {"A": "state A", "B": "state B"},
            "parents": {
                "atom_to_step": {"A": "P1-C01-S01", "B": "P2-C01-S01"},
                "step_to_cluster": {"P1-C01-S01": "P1-C01", "P2-C01-S01": "P2-C01"},
                "cluster_to_phase": {"P1-C01": "P1", "P2-C01": "P2"},
            },
            "priors": {
                "empirical_transitions": {
                    "A": {"A": 0.1, "B": 0.9},
                    "B": {"A": 0.1, "B": 0.9},
                },
                "procedural_transitions": {
                    "A": {"A": 0.1, "B": 0.9},
                    "B": {"A": 0.1, "B": 0.9},
                },
                "duration_distributions": {"A": {"1": 1.0}, "B": {"1": 1.0}},
                "duration_bin_size": 30.0,
                "smoothing_epsilon": 1e-6,
            },
        }

    def observation(self, time_sec, atom, boundary, a_prob, b_prob):
        phase = "P1" if atom == "A" else "P2"
        cluster = f"{phase}-C01"
        step = f"{cluster}-S01"
        return {
            "time_sec": time_sec,
            "atom": atom,
            "phase": phase,
            "cluster": cluster,
            "step": step,
            "confidence": max(a_prob, b_prob),
            "boundary_prob": boundary,
            "atom_topk": [{"atom": "A", "prob": a_prob}, {"atom": "B", "prob": b_prob}],
            "global_atom_probs": {"A": a_prob, "B": b_prob},
            "hierarchy_probs": {
                "phase": {"P1": a_prob, "P2": b_prob},
                "cluster": {"P1-C01": a_prob, "P2-C01": b_prob},
                "step": {"P1-C01-S01": a_prob, "P2-C01-S01": b_prob},
            },
        }

    def test_beliefs_are_normalized_and_delayed_commitment_switches(self):
        tracker = MarkovStateTracker(self.label_space)
        atom, first = tracker.step(self.observation(0.0, "A", 0.0, 0.99, 0.01))
        self.assertEqual(atom, "A")
        self.assertEqual(first["action"], "transition")

        scores = tracker.candidate_scores(self.observation(60.0, "B", 0.99, 0.01, 0.99))
        self.assertAlmostEqual(sum(item["belief"] for item in scores), 1.0)
        self.assertEqual(scores[0]["atom"], "B")
        self.assertIn("hierarchy", scores[0])

        atom, second = tracker.step(self.observation(60.0, "B", 0.99, 0.01, 0.99))
        self.assertEqual(atom, "B")
        self.assertEqual(second["action"], "transition")


class ObserverHierarchyTest(unittest.TestCase):
    def test_predicted_parent_mask_keeps_global_atom_distribution(self):
        label_space = {
            "levels": {
                "phase": ["P1", "P2"],
                "cluster": ["P1-C01", "P2-C01"],
                "step": ["P1-C01-S01", "P2-C01-S01"],
                "atom": ["A", "B"],
            },
            "node_names": {"A": "state A", "B": "state B"},
            "parents": {
                "atom_to_step": {"A": "P1-C01-S01", "B": "P2-C01-S01"},
                "step_to_cluster": {"P1-C01-S01": "P1-C01", "P2-C01-S01": "P2-C01"},
                "cluster_to_phase": {"P1-C01": "P1", "P2-C01": "P2"},
            },
        }
        observer = SurgMarkObserver(torch.nn.Identity(), None, label_space, hidden_size=2)
        logits = {
            "phase": torch.tensor([[5.0, 0.0]]),
            "cluster": torch.tensor([[0.0, 5.0]]),
            "step": torch.tensor([[0.0, 5.0]]),
            "atom": torch.tensor([[0.0, 5.0]]),
            "boundary": torch.tensor([0.0]),
        }

        state = observer.state_from_logits(logits, top_k=2)

        self.assertEqual(state["phase"], "P1")
        self.assertEqual(state["cluster"], "P1-C01")
        self.assertEqual(state["step"], "P1-C01-S01")
        self.assertEqual(state["atom"], "A")
        self.assertGreater(state["global_atom_probs"]["B"], state["global_atom_probs"]["A"])


if __name__ == "__main__":
    unittest.main()
