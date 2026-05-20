import sys
import zipfile
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_demo import package_assets_dir


def test_package_assets_dir_zips_asset_contents(tmp_path):
    assets_dir = tmp_path / "assets"
    nested_dir = assets_dir / "flow_matching"
    nested_dir.mkdir(parents=True)

    (assets_dir / "initial_frame.png").write_bytes(b"png")
    (nested_dir / "eval_results.txt").write_text("ok")

    archive_path = tmp_path / "assets_quick.zip"
    result = package_assets_dir(assets_dir, archive_path)

    assert result == archive_path
    assert archive_path.exists()

    with zipfile.ZipFile(archive_path) as zf:
        names = sorted(zf.namelist())

    assert names == [
        "flow_matching/eval_results.txt",
        "initial_frame.png",
    ]
