"""End-to-end inference with the frozen protocol artefact: raw photo -> src/preprocess.py ->
HOG -> pipeline. Usage: python predict.py <photo.png> <calib.json> [more photos...]"""
import json, pickle, sys, time
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
from src import preprocess as pp
import models as M

art = pickle.load(open(ROOT / "runs" / "protocol_v1" / "final_model.pkl", "rb"))
calib = json.loads(Path(sys.argv[2]).read_text())
for photo in [sys.argv[1], *sys.argv[3:]]:
    t = time.perf_counter()
    bgr = cv2.imread(photo, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        print(photo, "ERROR unreadable"); continue
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA if bgr.shape[2] == 3 else cv2.COLOR_BGRA2RGBA)
    step7 = pp.preprocess_stages(rgba, calib).resized          # same stage that is on disk
    p = art["pipeline"].predict_proba(M.hog_features(step7[None]))[0]
    ms = (time.perf_counter() - t) * 1000
    print(f"{Path(photo).parent.name}/{Path(photo).name}: {art['classes_hz'][p.argmax()]} Hz "
          f"(p={p.max():.2f}) {ms:.0f} ms")
