import torch

from profiles.backend.metrics import summarize


def test_summarize_reports_relative_l2_max_error_and_cosine():
    actual = torch.tensor([[[1.0, 2.0]]])
    reference = torch.tensor([[[1.0, 3.0]]])
    metrics = summarize(actual, reference)
    assert metrics["relative_l2"] == 0.3162277638912201
    assert metrics["max_absolute_error"] == 1.0
    assert metrics["cosine_similarity"] == 0.9899494647979736


def test_summarize_handles_zero_reference():
    actual = torch.zeros(1, 2, 2)
    reference = torch.zeros_like(actual)
    metrics = summarize(actual, reference)
    assert metrics["relative_l2"] == 0.0
    assert metrics["max_absolute_error"] == 0.0
