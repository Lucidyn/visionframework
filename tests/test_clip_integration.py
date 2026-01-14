import pytest


def test_clip_wrapper_smoke(monkeypatch):
    # Stub minimal torch/transformers functionality to exercise the wrapper.
    import types
    import sys

    class FakeTensor:
        def __init__(self, arr):
            import numpy as _np

            self._arr = _np.array(arr)

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def to(self, device):
            return self

    class FakeModel:
        def to(self, device):
            pass

        def eval(self):
            pass

        def get_image_features(self, **kwargs):
            return FakeTensor([[1.0, 0.0, 0.0]])

        def get_text_features(self, **kwargs):
            return FakeTensor([[1.0, 0.0, 0.0]])

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class FakeProcessor:
        def __call__(self, images=None, text=None, return_tensors=None, padding=None):
            if images is not None:
                return {"pixel_values": FakeTensor([[0]])}
            else:
                return {"input_ids": FakeTensor([[0]]), "attention_mask": FakeTensor([[1]])}

        @classmethod
        def from_pretrained(cls, name):
            return cls()

    fake_transformers = types.SimpleNamespace(CLIPProcessor=FakeProcessor, CLIPModel=FakeModel)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    class NoOpCtx:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.SimpleNamespace()
    fake_torch.tensor = lambda x: FakeTensor(x)
    fake_torch.no_grad = lambda: NoOpCtx()
    fake_torch.zeros = lambda shape, dtype=None: FakeTensor([[0] * (shape[1] if len(shape) > 1 else 1)])
    fake_torch.ones = lambda shape, dtype=None: FakeTensor([[1] * (shape[1] if len(shape) > 1 else 1)])

    class FakeAmp:
        def autocast(self):
            return NoOpCtx()

    fake_torch.cuda = types.SimpleNamespace(amp=FakeAmp())

    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Import the clip module directly by file path to avoid package-level imports
    import importlib.util
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    clip_path = repo_root / "visionframework" / "core" / "clip.py"
    spec = importlib.util.spec_from_file_location("vf_clip", str(clip_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    CLIPExtractor = mod.CLIPExtractor

    clip = CLIPExtractor({"device": "cpu"})
    clip.initialize()

    feats = clip.encode_text(["hello world"])  # returns numpy array
    import numpy as _np

    assert isinstance(feats, _np.ndarray)
    assert feats.shape[0] == 1
