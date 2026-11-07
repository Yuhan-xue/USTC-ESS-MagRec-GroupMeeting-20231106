import importlib.util
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "publication", ROOT / "tools/check_publication.py"
)
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


class PublicationTests(unittest.TestCase):
    def test_external_raw_location_has_an_explicit_missing_input_error(self):
        from magviz import Plane
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"MAG_RAW_DIR": directory}):
                with self.assertRaisesRegex(FileNotFoundError, "MAG_RAW_DIR"):
                    Plane("eq")

    def test_allowed_reduced_tables(self):
        for path in publication.ARRAY_KEYS:
            publication.validate_file(path, (ROOT / path).read_bytes())

    def test_raw_and_disguised_files_are_rejected(self):
        for path in (
            "data/eq/f00001.npz", "processed/coords.npz",
            "processed/raw.npy", "output/cache/recon_redo.pkl",
            "output/figs/cmp_tail.png", "raw.txt", "credentials.json",
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                publication.validate_file(path, b"")

    def test_2d_arrays_cannot_use_an_allowed_filename(self):
        content = io.BytesIO()
        np.savez(content, **{
            key: np.zeros((379, 2))
            for key in publication.ARRAY_KEYS["processed/eq_timeseries.npz"]
        })
        with self.assertRaises(ValueError):
            publication.validate_file("processed/eq_timeseries.npz", content.getvalue())

    def test_larger_or_incorrect_schema_is_rejected(self):
        content = io.BytesIO()
        np.savez(content, coordinates=np.zeros(3))
        with self.assertRaises(ValueError):
            publication.validate_file("processed/eq_timeseries.npz", content.getvalue())
        with self.assertRaises(ValueError):
            publication.validate_file("README.md", b"x" * (256 * 1024 + 1))

    def test_private_key_is_rejected(self):
        with self.assertRaises(ValueError):
            publication.validate_file("README.md", b"-----BEGIN " + b"PRIVATE KEY-----")

    def test_reconnection_counts_and_quadrants(self):
        files = {
            path: (ROOT / path).read_bytes()
            for path in (
                "processed/reconnection_candidates.csv",
                "processed/reconnection_summary.json",
            )
        }
        publication.validate_reconnection(files)


if __name__ == "__main__":
    unittest.main()
