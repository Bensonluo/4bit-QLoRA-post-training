"""Unit tests for DCP checkpoint utilities (src/training/distributed/checkpoint.py).

The roundtrip tests run *real* ``torch.distributed.checkpoint`` save/load in
no-dist CPU mode (small nn.Linear) — DCP is designed to work without a process
group, so this needs no GPUs and stays a unit test. Mocked tests cover the
async/save dispatch and availability branches.
"""

from __future__ import annotations

from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.training.distributed import checkpoint as dcp_utils


class TestDcpAvailable:
    def test_available_on_this_env(self) -> None:
        # torch 2.11 ships DCP — the cached probe must agree.
        assert dcp_utils.dcp_available() is True

    def test_result_is_cached(self) -> None:
        dcp_utils.dcp_available()
        assert dcp_utils._DCP_AVAILABLE is True
        # Second call returns the cached flag without re-importing.
        assert dcp_utils.dcp_available() is True

    def test_unavailable_when_import_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib.abc
        import sys

        monkeypatch.setattr(dcp_utils, "_DCP_AVAILABLE", None)
        # `import` consults sys.modules before any finder — evict the module so
        # the blocker is actually consulted (monkeypatch restores it after).
        monkeypatch.delitem(sys.modules, "torch.distributed.checkpoint", raising=False)

        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
                if fullname == "torch.distributed.checkpoint":
                    raise ImportError("blocked for test")
                return None

        blocker = _Blocker()
        sys.meta_path.insert(0, blocker)
        try:
            assert dcp_utils.dcp_available() is False
        finally:
            sys.meta_path.remove(blocker)
            monkeypatch.setattr(dcp_utils, "_DCP_AVAILABLE", None)  # reset probe


class TestUnavailableRaises:
    def test_save_raises_with_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dcp_utils, "_DCP_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="torch >= 2.4"):
            dcp_utils.save_dcp_checkpoint(nn.Linear(2, 2), "/tmp/x")
        monkeypatch.setattr(dcp_utils, "_DCP_AVAILABLE", True)

    def test_load_raises_with_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(dcp_utils, "_DCP_AVAILABLE", False)
        with pytest.raises(RuntimeError, match="torch >= 2.4"):
            dcp_utils.load_dcp_checkpoint(nn.Linear(2, 2), "/tmp/x")
        monkeypatch.setattr(dcp_utils, "_DCP_AVAILABLE", True)


class TestRealRoundtrip:
    """Real DCP save → load on CPU (no process group)."""

    def test_model_weights_roundtrip(self, tmp_path: object) -> None:
        model = nn.Linear(8, 8)
        ckpt = str(tmp_path) + "/ckpt"  # type: ignore[operator]

        dcp_utils.save_dcp_checkpoint(model, ckpt, asynchronous=False)

        fresh = nn.Linear(8, 8)
        with torch.no_grad():
            for p in fresh.parameters():
                p.copy_(torch.zeros_like(p))
        dcp_utils.load_dcp_checkpoint(fresh, ckpt)

        torch.testing.assert_close(
            dict(fresh.named_parameters())["weight"], dict(model.named_parameters())["weight"]
        )

    def test_extra_state_roundtrip(self, tmp_path: object) -> None:
        model = nn.Linear(4, 4)
        ckpt = str(tmp_path) + "/ckpt"  # type: ignore[operator]

        dcp_utils.save_dcp_checkpoint(
            model, ckpt, extra_state={"epoch": 7, "seed": 1234}, asynchronous=False
        )

        loaded = dcp_utils.load_dcp_checkpoint(
            nn.Linear(4, 4), ckpt, extra_state={"epoch": 0, "seed": 0}
        )
        assert loaded == {"epoch": 7, "seed": 1234}

    def test_optimizer_state_roundtrip(self, tmp_path: object) -> None:
        import torch.optim as optim

        model = nn.Linear(4, 4)
        opt = optim.SGD(model.parameters(), lr=0.1)
        model(torch.randn(2, 4)).sum().backward()
        opt.step()  # materialize optimizer state (momentum buffers)
        ckpt = str(tmp_path) + "/ckpt"  # type: ignore[operator]

        dcp_utils.save_dcp_checkpoint(model, ckpt, optimizer=opt, asynchronous=False)

        fresh = nn.Linear(4, 4)
        fresh_opt = optim.SGD(fresh.parameters(), lr=0.1)
        dcp_utils.load_dcp_checkpoint(fresh, ckpt, optimizer=fresh_opt)
        # Optimizer state (momentum) must have come back for every param.
        assert len(fresh_opt.state) == len(opt.state)


class TestDispatch:
    """Async vs sync dispatch and finalize — DCP calls mocked."""

    def _linear(self) -> nn.Linear:
        torch.manual_seed(0)
        return nn.Linear(4, 4)

    def test_async_returns_future(self) -> None:
        fut = Future()  # real type — the module isinstance-narrows the response
        with patch("torch.distributed.checkpoint.async_save", return_value=fut) as mas:
            out = dcp_utils.save_dcp_checkpoint(self._linear(), "/tmp/a")
        assert out is fut
        assert mas.call_args.kwargs["checkpoint_id"] == "/tmp/a"
        assert mas.call_args.kwargs["no_dist"] is True

    def test_async_response_normalized_to_upload_future(self) -> None:
        # A PROCESS checkpointer returns AsyncSaveResponse, not a Future —
        # the module normalizes to the upload_completion future.
        upload, response = MagicMock(), MagicMock()
        response.upload_completion = upload
        with patch("torch.distributed.checkpoint.async_save", return_value=response) as mas:
            out = dcp_utils.save_dcp_checkpoint(self._linear(), "/tmp/a")
        assert out is upload
        assert response.upload_completion is upload
        assert mas.call_args.kwargs["checkpoint_id"] == "/tmp/a"

    def test_sync_calls_save_and_returns_none(self) -> None:
        with patch("torch.distributed.checkpoint.save") as ms:
            out = dcp_utils.save_dcp_checkpoint(self._linear(), "/tmp/a", asynchronous=False)
        assert out is None
        assert ms.call_args.kwargs["no_dist"] is True
        state = ms.call_args.args[0]
        assert "model" in state and "optimizer" not in state

    def test_optimizer_included_in_state_template(self) -> None:
        import torch.optim as optim

        model = self._linear()
        opt = optim.SGD(model.parameters(), lr=0.1)
        with patch("torch.distributed.checkpoint.save") as ms:
            dcp_utils.save_dcp_checkpoint(model, "/tmp/a", optimizer=opt, asynchronous=False)
        assert "optimizer" in ms.call_args.args[0]

    def test_finalize_surfaces_failure(self) -> None:
        fut = MagicMock()
        fut.result.side_effect = OSError("disk full")
        with pytest.raises(OSError, match="disk full"):
            dcp_utils.finalize_dcp_save(fut)

    def test_finalize_none_is_noop(self) -> None:
        dcp_utils.finalize_dcp_save(None)
