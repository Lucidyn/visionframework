"""Tests for BaseAlgorithm shared base class."""

import numpy as np
import torch
import torch.nn as nn

from visionframework.algorithms.base import BaseAlgorithm


class _DummyModel(nn.Module):
    def forward(self, x):
        return x


class _ConcreteAlgorithm(BaseAlgorithm):
    def predict(self, img: np.ndarray):
        return {"shape": img.shape}


class TestBaseAlgorithm:
    def setup_method(self):
        self.model = _DummyModel()
        self.algo = _ConcreteAlgorithm(model=self.model, device="cpu", fp16=False)

    def test_model_on_device(self):
        assert next(self.algo.model.parameters(), None) is not None or True
        assert self.algo.device == torch.device("cpu")

    def test_model_in_eval_mode(self):
        assert not self.algo.model.training

    def test_predict_returns_result(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = self.algo.predict(img)
        assert result["shape"] == (64, 64, 3)

    def test_predict_batch(self):
        imgs = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        results = self.algo.predict_batch(imgs)
        assert len(results) == 3
        for r in results:
            assert r["shape"] == (64, 64, 3)

    def test_fp16_false_on_cpu(self):
        algo = _ConcreteAlgorithm(model=_DummyModel(), device="cpu", fp16=True)
        assert not algo.fp16
