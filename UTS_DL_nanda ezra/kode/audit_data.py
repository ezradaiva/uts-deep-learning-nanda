"""Inspect the supplied UTS archive and write a reproducible data audit."""
from __future__ import annotations

import argparse
import gzip
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def read_npz_member(archive: zipfile.ZipFile, member: str) -> dict[str, np.ndarray]:
    nested = zipfile.ZipFile(io.BytesIO(archive.read(member)))
    try:
        return {Path(name).stem: np.load(nested.open(name), allow_pickle=False) for name in nested.namelist()}
    finally:
        nested.close()


def read_gzip_csv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz:
        return pd.read_csv(gz)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Path to data_uts_dl.zip")
    parser.add_argument("--out", type=Path, default=Path("artifacts/data_audit.json"))
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    with zipfile.ZipFile(args.data) as z:
        dictionary = z.read("kamus_data.md").decode("utf-8")
        train_meta = pd.read_csv(z.open("klasifikasi_train_meta.csv"))
        test_meta = pd.read_csv(z.open("klasifikasi_test_meta.csv"))
        cluster_meta = pd.read_csv(z.open("clustering_meta.csv"))
        train_arrays = read_npz_member(z, "klasifikasi_train.npz")
        test_arrays = read_npz_member(z, "klasifikasi_test.npz")
        cluster_arrays = read_npz_member(z, "clustering.npz")
        rul_train = read_gzip_csv(z, "rul_train.csv.gz")
        rul_test = read_gzip_csv(z, "rul_test.csv.gz")

    sensor = train_arrays["sensor"].astype(np.float32)
    last_cycle = rul_train.groupby("id_unit")["siklus"].max()
    class_dist = train_meta["kelas"].value_counts().sort_index()
    by_line = pd.crosstab(train_meta["lini"], train_meta["kelas"])
    sensor_nan_sample = np.isnan(sensor).any(axis=(1, 2))
    report = {
        "archive": args.data.name,
        "datasheet": dictionary,
        "classification": {
            "train_rows": int(len(train_meta)),
            "test_rows": int(len(test_meta)),
            "train_classes": {str(k): int(v) for k, v in class_dist.items()},
            "line_class_counts": {str(i): {str(k): int(v) for k, v in row.items()} for i, row in by_line.iterrows()},
            "train_line_fraction": {str(k): float(v) for k, v in train_meta["lini"].value_counts(normalize=True).items()},
            "test_line_fraction": {str(k): float(v) for k, v in test_meta["lini"].value_counts(normalize=True).items()},
            "sensor_shape": list(sensor.shape),
            "sensor_nan_fraction": float(np.isnan(sensor).mean()),
            "sensor_rows_with_any_nan_fraction": float(sensor_nan_sample.mean()),
            "sensor_all_nan_rows": int(np.isnan(sensor).all(axis=(1, 2)).sum()),
            "image_shape": list(train_arrays["gambar"].shape),
            "test_image_shape": list(test_arrays["gambar"].shape),
        },
        "rul": {
            "train_rows": int(len(rul_train)),
            "test_rows": int(len(rul_test)),
            "train_machines": int(rul_train["id_unit"].nunique()),
            "test_machines": int(rul_test["id_unit"].nunique()),
            "train_missing_fraction": float(rul_train.isna().to_numpy().mean()),
            "test_missing_fraction": float(rul_test.isna().to_numpy().mean()),
            "train_last_cycle_quantiles": {str(k): float(v) for k, v in last_cycle.quantile([0, .25, .5, .75, 1]).items()},
            "train_sensor_std": {str(k): float(v) for k, v in rul_train.iloc[:, 5:].std().items()},
        },
        "clustering": {
            "samples": int(len(cluster_meta)),
            "microscope_counts": {str(k): int(v) for k, v in cluster_meta["mikroskop"].value_counts().items()},
            "image_shape": list(cluster_arrays["gambar"].shape),
            "spectrum_shape": list(cluster_arrays["spektrum"].shape),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "datasheet"}, indent=2, ensure_ascii=False))
    print(f"\nSaved audit: {args.out.resolve()}")


if __name__ == "__main__":
    main()
