"""Contact sheet: plate ROI (gray) and model input (edges) per class. Writes to runs/, never shows."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import preprocess as pp

def stages_for(path, session):
    calib = json.loads((ROOT / "data" / "calib" / f"{session}.json").read_text())
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA) if bgr.shape[2] == 3 else cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
    return pp.preprocess_stages(rgba, calib)

if __name__ == "__main__":
    out = ROOT / "runs" / "protocol_v1"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for session, hz in [("20261108-03", 497), ("20261108-03", 863), ("20261108-03", 1218), ("20261108-04", 2241)]:
        files = sorted((ROOT / "data/raw" / session / str(hz)).glob("IMG_*.png"))
        pick = [files[i] for i in (0, 7, 15, 23, len(files) - 1)]
        tiles = []
        for f in pick:
            s = stages_for(f, session)
            g = cv2.resize(s.gray, (192, 192)); e = cv2.resize(s.resized, (192, 192), interpolation=cv2.INTER_NEAREST)
            t = np.vstack([g, e]); cv2.putText(t, f"{hz} {f.stem[-4:]}", (4, 16), 0, 0.5, 255, 1); tiles.append(t)
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out / "contact_sheet.png"), np.vstack(rows))
    print("ok", out / "contact_sheet.png")
