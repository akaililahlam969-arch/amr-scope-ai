"""Génère une image synthétique d'antibiogramme pour tester le pipeline."""
import math
import sys

import cv2
import numpy as np

ZONES_MM = [22, 16, 14, 10]  # diamètres attendus (SXT, AMC, CIP, VA)


def make(path="sample_antibiogramme.jpg", seed=3):
    rng = np.random.default_rng(seed)
    S = 1000
    img = np.full((S, S, 3), (60, 55, 50), np.uint8)
    c = S // 2
    pr = 400
    scale = 2 * pr / 90.0
    cv2.circle(img, (c, c), pr, (150, 165, 175), -1)  # pelouse bactérienne (BGR)
    placements = []
    for k, d in enumerate(ZONES_MM):
        ang = math.radians(-135 + k * 90)
        cx = int(c + 205 * math.cos(ang))
        cy = int(c + 205 * math.sin(ang))
        placements.append((cx, cy))
        cv2.circle(img, (cx, cy), int(d / 2 * scale), (205, 215, 220), -1)  # zone claire
    for cx, cy in placements:
        cv2.circle(img, (cx, cy), int(3 * scale), (245, 245, 245), -1)  # disque de papier
        cv2.circle(img, (cx, cy), int(3 * scale), (200, 200, 200), 1)
    img = cv2.GaussianBlur(img, (0, 0), 2.2)
    noise = rng.normal(0, 3.5, img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)
    return path


if __name__ == "__main__":
    print(make(*(sys.argv[1:2] or [])))
