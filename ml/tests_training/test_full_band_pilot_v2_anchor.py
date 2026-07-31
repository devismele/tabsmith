r"""The pilot-v2 root anchor: what it constrains, and what it must leave alone.

The anchor exists because pilot v1 lost GuitarSet root accuracy while gaining
full-band accuracy. It is only a valid intervention if it (a) vanishes when the
student still agrees with the frozen teacher, (b) penalises drift away from it,
and (c) is confined to rehearsal frames, so it cannot suppress full-band
learning directly. Each of those is asserted here rather than assumed.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests_training -t .
"""
from __future__ import annotations

import unittest

import torch

from ml.full_band.pilot import root_distillation


class RootAnchorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260731)
        self.teacher = torch.randn(2, 5, 12)
        self.mask = torch.ones(2, 5)

    def test_identical_student_and_teacher_cost_nothing(self):
        anchor = root_distillation(self.teacher.clone(), self.teacher, self.mask)
        self.assertAlmostEqual(float(anchor), 0.0, places=6)

    def test_disagreement_is_penalised(self):
        student = torch.randn(2, 5, 12)
        anchor = root_distillation(student, self.teacher, self.mask)
        self.assertGreater(float(anchor), 0.0)

    def test_further_drift_costs_more(self):
        near = self.teacher + 0.05 * torch.randn(2, 5, 12)
        far = self.teacher + 2.0 * torch.randn(2, 5, 12)
        self.assertLess(float(root_distillation(near, self.teacher, self.mask)),
                        float(root_distillation(far, self.teacher, self.mask)))

    def test_it_is_invariant_to_a_shared_logit_shift(self):
        """A softmax anchor constrains the posterior, not the raw logit scale."""
        student = torch.randn(2, 5, 12)
        shifted = student + 3.7
        self.assertAlmostEqual(float(root_distillation(student, self.teacher, self.mask)),
                               float(root_distillation(shifted, self.teacher, self.mask)),
                               places=5)

    def test_masked_frames_do_not_contribute(self):
        student = self.teacher.clone()
        student[1] = student[1] + 5.0 * torch.randn(5, 12)
        rehearsal_only = torch.zeros(2, 5)
        rehearsal_only[0] = 1.0
        anchor = root_distillation(student, self.teacher, rehearsal_only)
        self.assertAlmostEqual(float(anchor), 0.0, places=6)

    def test_full_band_rows_are_excluded_from_the_average(self):
        """Row 1 stands in for a Slakh sample: it must not dilute or drive the term."""
        student = torch.randn(2, 5, 12)
        mask = torch.zeros(2, 5)
        mask[0] = 1.0
        masked = float(root_distillation(student, self.teacher, mask))
        row0_only = float(root_distillation(student[:1], self.teacher[:1], torch.ones(1, 5)))
        self.assertAlmostEqual(masked, row0_only, places=6)

    def test_padding_is_excluded(self):
        student = self.teacher.clone()
        student[:, 3:] = student[:, 3:] + 4.0
        mask = torch.ones(2, 5)
        mask[:, 3:] = 0.0
        self.assertAlmostEqual(float(root_distillation(student, self.teacher, mask)),
                               0.0, places=6)

    def test_an_empty_mask_does_not_divide_by_zero(self):
        student = torch.randn(2, 5, 12)
        anchor = root_distillation(student, self.teacher, torch.zeros(2, 5))
        self.assertTrue(torch.isfinite(anchor))
        self.assertAlmostEqual(float(anchor), 0.0, places=6)

    def test_it_is_differentiable_into_the_student_only(self):
        student = torch.randn(2, 5, 12, requires_grad=True)
        teacher = self.teacher.clone().requires_grad_(False)
        root_distillation(student, teacher, self.mask).backward()
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

    def test_non_positive_temperature_is_rejected(self):
        with self.assertRaises(ValueError):
            root_distillation(self.teacher, self.teacher, self.mask, temperature=0.0)


if __name__ == "__main__":
    unittest.main()
