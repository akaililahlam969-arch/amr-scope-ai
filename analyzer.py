"""AMR-Scope AI - module d'analyse d'image (OpenCV).

Pipeline :
  1. détection de la boîte de Pétri (HoughCircles) -> échelle px/mm (boîte de 90 mm)
  2. détection des disques d'antibiotique (HoughCircles, disque de 6 mm)
  3. mesure du diamètre d'inhibition par profil radial (médiane sur plusieurs angles)
  4. score de confiance + marquage "mesure incertaine" pour validation humaine
"""
from __future__ import annotations

import base64
import math

import cv2
import numpy as np

PLATE_MM = 90.0   # diamètre standard d'une boîte de Pétri
DISK_MM = 6.0     # diamètre standard d'un disque d'antibiotique
MAX_DISKS = 8
CONFIDENCE_THRESHOLD = 0.75  # en dessous -> validation humaine requise

# Ordre par défaut des étiquettes (modifiable par l'utilisateur à l'étape 3)
DEFAULT_LABELS = ["SXT-25", "AMC-30", "CIP-5", "VA-30", "CTX-30", "GEN-10", "AMP-10", "TET-30"]


def _resize(img: np.ndarray, max_side: int = 1000) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def detect_plate(gray: np.ndarray) -> tuple[float, float, float, bool]:
    """Retourne (cx, cy, r, trouvé)."""
    h, w = gray.shape
    blur = cv2.medianBlur(gray, 7)
    m = min(h, w)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.5, minDist=m // 2,
        param1=100, param2=45, minRadius=int(m * 0.25), maxRadius=int(m * 0.55),
    )
    if circles is None:
        return w / 2, h / 2, m * 0.45, False
    cx, cy, r = circles[0][0]
    return float(cx), float(cy), float(r), True


def detect_disks(gray: np.ndarray, plate: tuple[float, float, float]) -> list[tuple[float, float, float]]:
    cx, cy, pr = plate
    scale = (2 * pr) / PLATE_MM
    dr = DISK_MM / 2 * scale
    blur = cv2.GaussianBlur(gray, (5, 5), 1.2)
    found = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.0, minDist=dr * 3,
        param1=45, param2=16, minRadius=int(dr * 0.7), maxRadius=int(dr * 1.35),
    )
    out: list[tuple[float, float, float]] = []
    if found is None:
        return out
    for x, y, r in found[0]:
        if math.hypot(x - cx, y - cy) > pr * 0.88:
            continue  # trop près du bord
        if any(math.hypot(x - a, y - b) < dr * 2.5 for a, b, _ in out):
            continue
        # un vrai disque de papier est plus clair que son voisinage immédiat
        yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
        d2 = (xx - x) ** 2 + (yy - y) ** 2
        inside = gray[d2 <= (r * 0.7) ** 2].mean()
        ring = gray[(d2 >= (r * 1.4) ** 2) & (d2 <= (r * 2.0) ** 2)].mean()
        if inside - ring < 10:
            continue
        out.append((float(x), float(y), float(r)))
        if len(out) >= MAX_DISKS:
            break
    return out


def measure_zone(gray_smooth: np.ndarray, disk: tuple[float, float, float],
                 plate: tuple[float, float, float], others: list[tuple[float, float, float]],
                 scale: float) -> tuple[float, float]:
    """Diamètre (mm) de la zone d'inhibition + confiance [0..1]."""
    x, y, dr = disk
    pcx, pcy, pr = plate
    r_min = dr * 1.25
    # limite : 26 mm de rayon, bord de la boîte, et moitié de la distance au disque voisin
    r_max = min(26.0 * scale, pr - math.hypot(x - pcx, y - pcy) - 4)
    for ox, oy, _ in others:
        d = math.hypot(x - ox, y - oy)
        if d > 1:
            r_max = min(r_max, d * 0.5)
    if r_max <= r_min + 6:
        return DISK_MM, 0.0

    radii = np.arange(r_min, r_max, 0.5)
    edges, strengths = [], []
    for ang in np.deg2rad(np.arange(0, 360, 8)):
        xs = x + radii * np.cos(ang)
        ys = y + radii * np.sin(ang)
        prof = cv2.remap(gray_smooth, xs.astype(np.float32).reshape(1, -1),
                         ys.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel().astype(np.float32)
        prof = np.convolve(prof, np.ones(5) / 5, mode="same")
        grad = np.gradient(prof)
        grad[:4] = 0
        grad[-4:] = 0
        i = int(np.argmax(np.abs(grad)))
        # contraste local entre intérieur de la zone et pelouse
        inner = prof[max(0, i - 8):max(1, i - 2)].mean() if i > 2 else prof[0]
        outer = prof[min(len(prof) - 1, i + 3):min(len(prof), i + 10)].mean()
        edges.append(radii[i])
        strengths.append(abs(outer - inner))

    edges_a = np.array(edges)
    strengths_a = np.array(strengths)
    keep = strengths_a >= max(4.0, np.percentile(strengths_a, 30))
    if keep.sum() < 8:
        return DISK_MM, 0.0
    e = edges_a[keep]
    r_med = float(np.median(e))
    spread = float(np.median(np.abs(e - r_med))) / max(r_med, 1)  # dispersion relative
    contrast = float(np.clip(np.median(strengths_a[keep]) / 20.0, 0, 1))
    coverage = keep.sum() / len(edges_a)
    at_limit = abs(r_med - radii[-1]) < 2.0
    conf = (1 - min(spread * 4, 1)) * 0.5 + contrast * 0.3 + coverage * 0.2
    if at_limit:
        conf *= 0.5
    return 2 * r_med / scale, float(np.clip(conf, 0, 1))


def analyze(image_bgr: np.ndarray) -> dict:
    img, _ = _resize(image_bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pcx, pcy, pr, plate_ok = detect_plate(gray)
    plate = (pcx, pcy, pr)
    scale = (2 * pr) / PLATE_MM
    disks = detect_disks(gray, plate)

    # ordre horaire à partir de la gauche-haut pour des étiquettes reproductibles
    disks.sort(key=lambda d: (math.atan2(d[1] - pcy, d[0] - pcx) + math.pi * 1.25) % (2 * math.pi))

    smooth = cv2.GaussianBlur(gray, (0, 0), 2.0)
    results = []
    annotated = img.copy()
    cv2.circle(annotated, (int(pcx), int(pcy)), int(pr), (255, 120, 0), 2)
    for idx, d in enumerate(disks):
        others = [o for j, o in enumerate(disks) if j != idx]
        diam, conf = measure_zone(smooth, d, plate, others, scale)
        uncertain = conf < CONFIDENCE_THRESHOLD
        label = DEFAULT_LABELS[idx % len(DEFAULT_LABELS)]
        results.append({
            "id": idx + 1,
            "label": label,
            "x": round(d[0] / img.shape[1], 4),
            "y": round(d[1] / img.shape[0], 4),
            "diameter_mm": round(diam, 1),
            "confidence": round(conf, 2),
            "uncertain": bool(uncertain),
        })
        color = (0, 140, 255) if uncertain else (60, 180, 60)
        cv2.circle(annotated, (int(d[0]), int(d[1])), int(d[2]), (255, 255, 255), 2)
        cv2.circle(annotated, (int(d[0]), int(d[1])), int(diam / 2 * scale), color, 2)
        cv2.putText(annotated, f"{label} {diam:.0f}mm", (int(d[0]) - 40, int(d[1]) - int(d[2]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {
        "plate_detected": plate_ok,
        "scale_px_per_mm": round(scale, 2),
        "disk_count": len(results),
        "disks": results,
        "needs_review": any(r["uncertain"] for r in results) or not plate_ok or not results,
        "annotated": "data:image/jpeg;base64," + base64.b64encode(buf).decode(),
    }
