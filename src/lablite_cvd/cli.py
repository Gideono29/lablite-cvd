"""Command-line entry point: ``lablite-cvd {download,cohort}``."""
import argparse
from pathlib import Path

from lablite_cvd.pipeline.config import DEFAULT_DATA_DIR


def main(argv=None):
    p = argparse.ArgumentParser(prog="lablite-cvd", description="LabLite-CVD data pipeline (research use only)")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download", help="Download NHANES 1999-2018 and NCHS 2019 linked mortality files")
    dl.add_argument("--jobs", type=int, default=4)
    sub.add_parser("cohort", help="Build data/processed/cohort.csv.gz with missing-lab indicators")
    ft = sub.add_parser("fit", help="Fit tier models and estimate the information cost of missing labs")
    ft.add_argument("--out", type=Path, default=Path("outputs/fit"))
    ft.add_argument("--horizon", type=float, default=10.0)
    args = p.parse_args(argv)

    if args.cmd == "download":
        from lablite_cvd.pipeline.download import download_all
        download_all(args.data_dir, jobs=args.jobs)
    elif args.cmd == "cohort":
        from lablite_cvd.pipeline.cohort import build_cohort
        build_cohort(args.data_dir)
    elif args.cmd == "fit":
        from lablite_cvd.pipeline.fit import run_fit
        run_fit(args.data_dir, args.out, horizon=args.horizon)


if __name__ == "__main__":
    main()
