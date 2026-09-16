import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_metric_adapter.py"
SPEC = importlib.util.spec_from_file_location("train_metric_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MetricAdapterTests(unittest.TestCase):
    def test_scheduler_horizon_can_match_full_training_during_screening(self):
        self.assertEqual(MODULE.resolve_scheduler_steps(500, None), 500)
        self.assertEqual(MODULE.resolve_scheduler_steps(500, 2000), 2000)
        with self.assertRaises(ValueError):
            MODULE.resolve_scheduler_steps(500, 499)

    def test_batch_hard_triplet_prefers_separated_identities(self):
        labels = torch.tensor([0, 0, 1, 1])
        separated = torch.tensor([
            [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0],
        ])
        confused = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0],
        ])

        separated_loss = MODULE.batch_hard_triplet_loss(separated, labels, margin=0.2)
        confused_loss = MODULE.batch_hard_triplet_loss(confused, labels, margin=0.2)

        self.assertLess(separated_loss.item(), confused_loss.item())

    def test_batch_hard_triplet_is_finite_and_backpropagates(self):
        raw_features = torch.randn(6, 4, requires_grad=True)
        features = torch.nn.functional.normalize(raw_features, dim=1)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])

        loss = MODULE.batch_hard_triplet_loss(features, labels, margin=0.2)

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(raw_features.grad)

    def test_checkpoint_builder_restores_residual_mode(self):
        source = MODULE.ResidualMetricAdapter(4, 8, 4, residual=True)
        for parameter in source.project.parameters():
            torch.nn.init.zeros_(parameter)
        checkpoint = {"model": source.state_dict(), "config": {"residual": True}}

        restored = MODULE.build_adapter_from_checkpoint(checkpoint)
        output = restored(torch.tensor([[3.0, 0.0, 4.0, 0.0]]))

        torch.testing.assert_close(output, torch.tensor([[0.6, 0.0, 0.8, 0.0]]))

    def test_groups_respect_requested_images_per_identity(self):
        labels = np.asarray(["a", "a", "b", "b", "b", "c"])

        eligible, groups = MODULE.make_groups(labels, min_images=3)

        self.assertEqual(eligible, ["b"])
        self.assertEqual(groups["b"].tolist(), [2, 3, 4])

    def test_residual_adapter_keeps_normalized_input_when_projection_is_zero(self):
        model = MODULE.ResidualMetricAdapter(4, 8, 4, residual=True)
        for parameter in model.project.parameters():
            torch.nn.init.zeros_(parameter)
        vectors = torch.tensor([[3.0, 0.0, 4.0, 0.0]])

        output = model(vectors)

        torch.testing.assert_close(output, torch.tensor([[0.6, 0.0, 0.8, 0.0]]))

    def test_balanced_batch_has_two_examples_per_identity(self):
        vectors = np.arange(48, dtype=np.float32).reshape(12, 4)
        labels = np.asarray(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3)
        eligible, groups = MODULE.make_groups(labels)
        batch, numeric = MODULE.sample_balanced_batch(
            vectors, eligible, groups, identities_per_batch=3, images_per_identity=2,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(tuple(batch.shape), (6, 4))
        self.assertEqual(torch.bincount(numeric).tolist(), [2, 2, 2])

    def test_contrastive_loss_is_finite_and_backpropagates(self):
        model = MODULE.ResidualMetricAdapter(4, 8, 3)
        vectors = torch.randn(6, 4)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = MODULE.supervised_contrastive_loss(model(vectors), labels, 0.1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
