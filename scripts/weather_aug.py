"""Adverse-weather + camera-degradation augmentation pipeline.

Generates, for a clean YOLO-format dataset, weather variants of a controlled
fraction of TRAIN images only (validation/test stay clean/real).

Effects (randomly combined with capped intensity):
  - rain streaks           (directional lines + slight brightness drop)
  - fog / haze             (additive veil, contrast drop)
  - low light / night      (gamma down + blue shift)
  - glare / sun glare      (localized bright blob)
  - motion blur            (horizontal/diagonal kernel)
  - defocus blur           (gaussian)
  - sensor noise           (gaussian + salt-pepper)
  - jpeg compression       (quality 15-40)
  - water droplets on lens (sparse blobs) - light intensity only

Each effect has a probability and intensity range; the final image is the
composition of 1-3 randomly chosen effects (never all at once).
"""
import random
import math
import cv2
import numpy as np


# ---------------------------------------------------------------- effects
def rain(img, intensity=0.5):
    h, w = img.shape[:2]
    layer = np.zeros((h, w), dtype=np.uint8)
    n_drops = int(w * h * 0.0004 * (0.5 + intensity))
    length = int(10 + 18 * intensity)
    angle = math.radians(random.uniform(75, 105))  # mostly vertical streaks
    dx, dy = math.cos(angle), math.sin(angle)
    for _ in range(n_drops):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = int(x + dx * length), int(y + dy * length)
        shade = random.randint(140, 220)
        cv2.line(layer, (x, y), (x2, y2), shade, 1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    streaks = cv2.cvtColor(layer, cv2.COLOR_GRAY2BGR).astype(np.float32)
    out = cv2.addWeighted(img.astype(np.float32), 1.0, streaks, 0.45 * intensity + 0.15, 0)
    # slight overall darkening / saturation drop (wet look)
    out *= random.uniform(0.75, 0.95)
    return np.clip(out, 0, 255).astype(np.uint8)


def fog(img, intensity=0.5):
    h, w = img.shape[:2]
    veil = np.full_like(img, random.randint(190, 235), dtype=np.float32)
    # depth-ish gradient: stronger haze toward top (farther) or one side
    grad = np.linspace(0.55, 1.0, h, dtype=np.float32)[:, None, None]
    veil *= grad
    out = cv2.addWeighted(img.astype(np.float32), 1.0 - 0.55 * intensity,
                          veil, 0.55 * intensity, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def low_light(img, intensity=0.5):
    gamma = 1.0 + 1.4 * intensity          # >1 darkens
    table = np.array([(i / 255.0) ** gamma * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(img, table)
    # bluish night tint
    blue_tint = np.array([0.9, 0.95, 1.12], dtype=np.float32)
    out = np.clip(out.astype(np.float32) * blue_tint, 0, 255).astype(np.uint8)
    return out


def glare(img, intensity=0.5):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    cx, cy = random.randint(w // 5, 4 * w // 5), random.randint(h // 5, 3 * h // 5)
    radius = int(min(h, w) * (0.15 + 0.25 * intensity))
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (cx, cy), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), radius / 2.5)
    strength = 90 + 110 * intensity
    out += mask[..., None] * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def motion_blur(img, intensity=0.5):
    k = int(6 + 22 * intensity)
    if k % 2 == 0:
        k += 1
    angle = random.uniform(0, math.pi)
    kernel = np.zeros((k, k), dtype=np.float32)
    c = k // 2
    dx, dy = math.cos(angle) * c, math.sin(angle) * c
    cv2.line(kernel, (int(c - dx), int(c - dy)), (int(c + dx), int(c + dy)), 1.0, 1)
    kernel /= kernel.sum()
    return cv2.filter2D(img, -1, kernel)


def defocus_blur(img, intensity=0.5):
    sigma = 0.8 + 2.8 * intensity
    return cv2.GaussianBlur(img, (0, 0), sigma)


def sensor_noise(img, intensity=0.5):
    out = img.astype(np.float32)
    sigma = 6 + 26 * intensity
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out += noise
    # sparse salt & pepper
    n_sp = int(img.shape[0] * img.shape[1] * 0.0008 * (0.5 + intensity))
    if n_sp:
        ys = np.random.randint(0, img.shape[0], n_sp)
        xs = np.random.randint(0, img.shape[1], n_sp)
        vals = np.random.choice([0, 255], n_sp)
        out[ys, xs, :] = vals[:, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def jpeg_artifacts(img, intensity=0.5):
    q = int(65 - 45 * intensity)  # quality 65 down to 20
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, max(q, 10)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img


def droplets(img, intensity=0.5):
    """Simulates water drops on the lens - used sparingly."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    n = int(8 + 30 * intensity)
    for _ in range(n):
        r = random.randint(3, 10)
        x, y = random.randint(0, w - r), random.randint(0, h - r)
        blob = np.zeros((h, w), np.float32)
        cv2.circle(blob, (x, y), r, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r / 2)
        out = out * (1 - blob[..., None] * 0.5) + blob[..., None] * 255 * 0.5
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- pipeline
# (function, probability, (min_intensity, max_intensity))
AUGMENTATIONS = [
    (rain,           0.16, (0.3, 0.9)),
    (fog,            0.16, (0.3, 0.9)),
    (low_light,      0.16, (0.3, 0.9)),
    (motion_blur,    0.12, (0.3, 0.9)),
    (defocus_blur,   0.10, (0.3, 0.8)),
    (sensor_noise,   0.12, (0.3, 0.9)),
    (glare,          0.08, (0.3, 0.8)),
    (jpeg_artifacts, 0.08, (0.4, 0.9)),
    (droplets,       0.04, (0.2, 0.5)),
]


def augment_image(img, rng=None):
    """Apply 1-3 random weather/camera effects, capped intensity."""
    rng = rng or random
    chosen = [a for a in AUGMENTATIONS if rng.random() < a[1]]
    if not chosen:
        chosen = [rng.choice(AUGMENTATIONS)]
    rng.shuffle(chosen)
    chosen = chosen[:3]
    out = img
    for fn, _, irange in chosen:
        lo, hi = irange
        intensity = rng.uniform(lo, hi)
        out = fn(out, intensity)
    return out
