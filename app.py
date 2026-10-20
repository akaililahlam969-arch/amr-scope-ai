"""AMR-Scope AI - serveur Flask.  Lancer :  python app.py  puis ouvrir http://localhost:5000"""
import datetime
import json
import os
import sqlite3
from contextlib import closing

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from analyzer import analyze

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
IMG_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "amr.db")
os.makedirs(IMG_DIR, exist_ok=True)


def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


with closing(db()) as _c:
    _c.execute("""CREATE TABLE IF NOT EXISTS analyses(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        site TEXT,
        needs_review INTEGER NOT NULL,
        results TEXT NOT NULL)""")
    _c.commit()


def seq_of(row_id: int, created_at: str) -> str:
    return f"AMR-{created_at[:4]}-{row_id:04d}"

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024

# Seuils de démonstration (mm) : S si diamètre >= s, R si < r, sinon I.
# ATTENTION : les vrais seuils EUCAST dépendent de l'espèce bactérienne et de la version
# du tableau. À remplacer / valider avant tout usage réel.
BREAKPOINTS = {
    "SXT-25": {"s": 14, "r": 11},
    "AMC-30": {"s": 19, "r": 14},
    "CIP-5": {"s": 25, "r": 22},
    "VA-30": {"s": 17, "r": 17},
    "CTX-30": {"s": 20, "r": 17},
    "GEN-10": {"s": 17, "r": 17},
    "AMP-10": {"s": 14, "r": 14},
    "TET-30": {"s": 19, "r": 17},
}


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/api/breakpoints")
def breakpoints():
    return jsonify(BREAKPOINTS)


@app.post("/api/analyze")
def api_analyze():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="Aucune image reçue."), 400
    data = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Image illisible."), 400
    return jsonify(analyze(img))


@app.post("/api/save")
def api_save():
    """Enregistre une analyse validée (image originale + résultats + site)."""
    try:
        results = json.loads(request.form.get("results", "[]"))
        assert isinstance(results, list) and results
        clean = []
        for r in results:
            label = str(r["label"])[:30]
            mm = float(r["mm"])
            cls = r["r"]
            assert cls in ("S", "I", "R", "?") and 0 <= mm <= 80
            clean.append({"label": label, "mm": round(mm, 1), "r": cls})
    except Exception:
        return jsonify(error="Résultats invalides."), 400

    img = None
    f = request.files.get("image")
    if f is not None:
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(error="Image illisible."), 400

    site = request.form.get("site", "").strip()[:200]
    needs = 1 if request.form.get("needs_review") == "1" else 0
    now = datetime.datetime.now().isoformat(timespec="seconds")
    with closing(db()) as con:
        cur = con.execute(
            "INSERT INTO analyses(created_at, site, needs_review, results) VALUES (?,?,?,?)",
            (now, site, needs, json.dumps(clean)))
        con.commit()
        rid = cur.lastrowid
    if img is not None:
        cv2.imwrite(os.path.join(IMG_DIR, f"{rid}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return jsonify(id=rid, seq=seq_of(rid, now), created_at=now)


@app.get("/api/history")
def api_history():
    with closing(db()) as con:
        rows = con.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT 100").fetchall()
    out = []
    for r in rows:
        res = json.loads(r["results"])
        out.append({
            "id": r["id"], "seq": seq_of(r["id"], r["created_at"]),
            "created_at": r["created_at"], "site": r["site"],
            "needs_review": bool(r["needs_review"]), "results": res,
            "has_image": os.path.exists(os.path.join(IMG_DIR, f"{r['id']}.jpg")),
        })
    return jsonify(out)


@app.get("/api/image/<int:rid>")
def api_image(rid):
    return send_from_directory(IMG_DIR, f"{rid}.jpg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
