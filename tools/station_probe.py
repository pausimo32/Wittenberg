#!/usr/bin/env python3
"""Feasibility probe for the capture station — run once, on the MacBook, iPhone attached.

    .venv/bin/python tools/station_probe.py            # then open the printed URL

`docs/adr/0008` is *proposed* until this has run: every figure it marks *computed* or
*unknown* is a question one of the page's blocks answers on the real hardware. The answers
are saved, as one JSON file, to `runs/station-probe-<ts>/probe.json`.

**Throwaway, and outside `site/` on purpose.** It is not part of the app and nothing in the
app imports it. Once the ADR is settled, delete `tools/station_probe*`.

**It owns no gate and no storage.** It mounts `dataset_api.dataset_router` unchanged — the
routes `/dades` and the station drive — so the calibration it reports is `calibrate()`'s and
the verdict it reports is `ingest.assess_frame`'s. What it does differently is *where*: both
roots live under its own `runs/station-probe-<ts>/`, so a probe frame can never reach the
real `data/` or `drafts/`, and the run directory keeps the calibration frame for review.

It does not go through `api.create_app`, which refuses to start without a trained run — and
there is none yet: the station exists to collect the data the first real one is trained on.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from calibrate import load_calibration_config  # noqa: E402
from dataset_api import DatasetState, dataset_router  # noqa: E402
from preprocess import load_config  # noqa: E402
from quality import load_gate  # noqa: E402

HOST = "127.0.0.1"
PORT = 8765
PAGE_DIR = ROOT / "tools" / "station_probe"


def build_app(run_dir: Path) -> FastAPI:
    app = FastAPI(title="Wittenberg station probe")
    calibration_config = load_calibration_config()

    @app.get("/probe/context")
    def context() -> dict[str, Any]:
        """What the page cannot know by itself: the rig convention it is measured against."""
        return {
            "run_dir": str(run_dir),
            "calibration_config": calibration_config,
            "today": datetime.now().strftime("%Y%m%d"),
        }

    @app.post("/probe/report")
    def report(body: dict[str, Any]) -> dict[str, str]:
        path = run_dir / "probe.json"
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False), "utf-8")
        return {"path": str(path)}

    app.include_router(
        dataset_router(
            DatasetState(
                data_root=run_dir / "data",
                drafts_root=run_dir / "drafts",
                calibration_config=calibration_config,
                preprocess_config=load_config(),
                gate=load_gate(),
            )
        )
    )
    # Last, so it cannot shadow /api or /probe.
    app.mount("/", StaticFiles(directory=PAGE_DIR, html=True), name="probe")
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)

    run_dir = ROOT / "runs" / f"station-probe-{datetime.now():%Y%m%d-%H%M%S}"
    (run_dir / "data").mkdir(parents=True)
    (run_dir / "drafts").mkdir()

    import uvicorn

    print(f"  station probe -> {run_dir.relative_to(ROOT)}")
    print(f"  http://{HOST}:{args.port}  — 127.0.0.1 only, no outbound call")
    uvicorn.run(build_app(run_dir), host=HOST, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
