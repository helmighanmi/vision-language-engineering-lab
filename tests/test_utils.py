# Path: tests/test_utils.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from pathlib import Path

from vlm_engineering.utils import extract_json_object, to_image_reference


def test_extract_json_object_from_fenced_output() -> None:
    assert extract_json_object('```json\n{"ok": true}\n```') == {"ok": True}


def test_to_image_reference_for_local_file(tmp_path: Path) -> None:
    image = tmp_path / "x.png"
    image.write_bytes(b"x")
    assert to_image_reference(image).startswith("file://")
