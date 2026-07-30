import pytest
import torch

from src.models.vggt import VGGTModel


class _FakeAggregator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(self, value):
        return [self.projection(value)], 7


class _FakeVGGT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.aggregator = _FakeAggregator()
        self.head = torch.nn.Linear(2, 2)


def test_mixed_float16_casts_only_aggregator_and_restores_float_tokens():
    adapter = VGGTModel(
        {"device": "cuda", "weights_dtype": "mixed_float16"}
    )
    adapter.model = _FakeVGGT()

    adapter._configure_model_precision()
    tokens, patch_start_idx = adapter.model.aggregator(
        torch.ones((1, 2), dtype=torch.float16)
    )

    assert adapter.model.aggregator.projection.weight.dtype == torch.float16
    assert adapter.model.head.weight.dtype == torch.float32
    assert tokens[0].dtype == torch.float32
    assert patch_start_idx == 7


def test_mixed_float16_rejects_cpu_device():
    adapter = VGGTModel(
        {"device": "cpu", "weights_dtype": "mixed_float16"}
    )
    adapter.model = _FakeVGGT()

    with pytest.raises(ValueError, match="require a CUDA device"):
        adapter._configure_model_precision()


def test_unknown_weights_dtype_is_rejected():
    adapter = VGGTModel({"device": "cuda", "weights_dtype": "float8"})
    adapter.model = _FakeVGGT()

    with pytest.raises(ValueError, match="Invalid weights_dtype"):
        adapter._configure_model_precision()
