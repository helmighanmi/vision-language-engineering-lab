# Path: tests/test_qwen_loader.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from pathlib import Path

from vlm_engineering.qwen import QwenVLModel


def test_local_loader_is_offline(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model = QwenVLModel.from_local(model_dir)
    assert model.local_files_only is True
    assert model.trust_remote_code is False
