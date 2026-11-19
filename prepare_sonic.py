"""
prepare_sonic.py
================
Pré-traitement (à lancer une seule fois) :
  1. Détourage chroma-key des vidéos sur fond vert -> sprites RGBA rognés
  2. Analyse de l'animation (cycle de marche, frame d'impact, position de la main)
  3. Téléchargement des modèles MediaPipe (visage + segmentation de la personne)

Personnages :
  sonic  : "Sonic Slap Greenscreen.mp4"   -> marche + claque
  hornet : "Hornet Dance Greenscreen.mp4" -> passage seulement (danse en tournant sur elle-même)

Sortie : assets/<perso>/frame_XXX.png + assets/<perso>/meta.json + assets/models/*.tflite

Usage :  python prepare_sonic.py            (tous les personnages)
         python prepare_sonic.py hornet     (un seul)
"""
import json
import os
import sys
import urllib.request

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")
MODELS = os.path.join(ASSETS, "models")

MODEL_URLS = {
    "blaze_face_short_range.tflite":
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "selfie_segmenter.tflite":
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
        "selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}

# --- Sonic : découpage analysé sur la vidéo fournie (224 frames @ 29.97 fps)
SONIC_SRC = os.path.join(ROOT, "Sonic Slap Greenscreen.mp4")
WALK_LOOP = (0, 36)      # cycle de marche vers la droite, boucle parfaite de 36 frames
SLAP_SEQ = (144, 224)    # 144 == début de cycle -> enchaînement sans saut ; arrêt, demi-tour, claque
IMPACT_SEARCH = (192, 204)  # la main balaie vers la gauche ici

# --- Hornet : danse sur place (139 frames @ 29.97 fps, 1 tour complet ≈ 18 frames).
# La frame 110 est quasi identique à la 19 : 19..109 forme une boucle parfaite de 5 tours.
# Seule cette boucle est gardée (renumérotée 0..90) : pas de claque, uniquement le passage.
HORNET_SRC = os.path.join(ROOT, "Hornet Dance Greenscreen.mp4")
HORNET_LOOP = (19, 110)


def key_frame(bgr):
    """Chroma key fond vert -> (BGR despillé, alpha float 0..1)."""
    f = bgr.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    greenness = g - np.maximum(r, b)          # ~100 sur le fond, <= 0 sur le perso
    alpha = 1.0 - np.clip((greenness - 18.0) / (55.0 - 18.0), 0.0, 1.0)

    # nettoyage : on garde le gros personnage, on vire les poussières de compression
    hard = (alpha > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
    if n > 1:
        keep = np.zeros(n, bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep[1:] = areas >= max(150, 0.01 * areas.max())
        alpha *= cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)

    # bord légèrement rentré + adouci (évite le liseré vert)
    a8 = (alpha * 255).astype(np.uint8)
    a8 = cv2.erode(a8, np.ones((3, 3), np.uint8))
    a8 = cv2.GaussianBlur(a8, (3, 3), 0)
    alpha = a8.astype(np.float32) / 255.0

    # despill : le vert ne doit jamais dépasser max(R, B)
    g2 = np.minimum(g, np.maximum(r, b) + 4.0)
    out = np.stack([b, g2, r], -1).clip(0, 255).astype(np.uint8)
    return out, alpha


def extract(src, out_dir, keep=None):
    """Détoure la vidéo et écrit les sprites rognés. keep = (début, fin) : frames gardées, renumérotées dès 0.
    -> (frames [{ox, oy, w, h, alpha}], fps, largeur, hauteur)"""
    if not os.path.exists(src):
        sys.exit(f"Vidéo introuvable : {src}")
    os.makedirs(out_dir, exist_ok=True)
    for old in os.listdir(out_dir):                       # pas de vieilles frames en trop
        if old.startswith("frame_") and old.endswith(".png"):
            os.remove(os.path.join(out_dir, old))
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    i = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        i += 1
        if keep and not keep[0] <= i < keep[1]:
            continue
        rgb, a = key_frame(bgr)
        ys, xs = np.where(a > 0.02)
        if len(xs) == 0:
            x0 = y0 = 0
            x1 = y1 = 1
        else:
            x0, x1 = max(0, xs.min() - 2), min(W, xs.max() + 3)
            y0, y1 = max(0, ys.min() - 2), min(H, ys.max() + 3)
        rgba = np.dstack([rgb, (a * 255).astype(np.uint8)])[y0:y1, x0:x1]
        cv2.imwrite(os.path.join(out_dir, f"frame_{len(frames):03d}.png"), rgba)
        frames.append({"ox": int(x0), "oy": int(y0), "w": int(x1 - x0), "h": int(y1 - y0), "alpha": a})
    cap.release()
    print(f"{len(frames)} frames détourées ({W}x{H} @ {fps:.2f} fps) -> {out_dir}")
    return frames, fps, W, H


def ground_ref(frames, loop):
    """Point de référence (centre des pieds au sol) et hauteur du perso, mesurés sur la boucle."""
    sel = frames[loop[0]:loop[1]]
    ground_y = float(np.percentile([f["oy"] + f["h"] for f in sel], 90))
    ref_x = float(np.mean([np.where(f["alpha"] > 0.5)[1].mean() for f in sel]))
    char_h = float(ground_y - np.median([f["oy"] for f in sel]))   # hauteur "perso" (tête -> pieds)
    return ref_x, ground_y, char_h


def write_meta(out_dir, meta, frames):
    meta["frames"] = [{k: f[k] for k in ("ox", "oy", "w", "h")} for f in frames]
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=1)
    print("OK ->", out_dir)


def prepare_sonic():
    out = os.path.join(ASSETS, "sonic")
    frames, fps, W, H = extract(SONIC_SRC, out)
    ref_x, ground_y, char_h = ground_ref(frames, WALK_LOOP)

    # --- frame d'impact : la main la plus à gauche, à hauteur de tête/épaules
    best = None
    for k in range(*IMPACT_SEARCH):
        ys, xs = np.where(frames[k]["alpha"] > 0.5)
        band = ys < (ground_y - 0.35 * char_h)   # au-dessus du bassin
        if not band.any():
            continue
        xmin = xs[band].min()
        if best is None or xmin < best[1]:
            sel = band & (xs <= xmin + 25)
            best = (k, xmin, float(xs[sel].mean()), float(ys[sel].mean()))
    impact_frame, _, hx, hy = best
    print(f"impact frame {impact_frame}, main en ({hx:.0f},{hy:.0f}), ref pieds ({ref_x:.0f},{ground_y:.0f}), "
          f"hauteur perso {char_h:.0f}px")

    write_meta(out, {
        "name": "Sonic",
        "fps": fps, "src_w": W, "src_h": H, "count": len(frames),
        "ref": [ref_x, ground_y], "char_h": char_h,
        "walk_loop": list(WALK_LOOP), "slap_seq": list(SLAP_SEQ),
        "impact_frame": int(impact_frame), "hand": [hx, hy],
        # Sonic regarde vers la droite en marchant, et claque vers la GAUCHE
        "walk_faces": "right", "slap_dir": "left",
    }, frames)


def prepare_hornet():
    out = os.path.join(ASSETS, "hornet")
    frames, fps, W, H = extract(HORNET_SRC, out, HORNET_LOOP)
    loop = (0, len(frames))
    ref_x, ground_y, char_h = ground_ref(frames, loop)
    print(f"ref pieds ({ref_x:.0f},{ground_y:.0f}), hauteur perso {char_h:.0f}px")
    write_meta(out, {
        "name": "Hornet",
        "fps": fps, "src_w": W, "src_h": H, "count": len(frames),
        "ref": [ref_x, ground_y], "char_h": char_h,
        "walk_loop": list(loop),
        "slap_seq": None,                  # pas de claque : uniquement le passage
        "source_frames": list(HORNET_LOOP),
        # elle tourne sur elle-même : on la fait glisser à vitesse de danse, pas de foulée à respecter
        "stride": 0.03, "walk_speed": [1.0, 1.15],
    }, frames)


def download_models():
    os.makedirs(MODELS, exist_ok=True)
    for name, url in MODEL_URLS.items():
        p = os.path.join(MODELS, name)
        if os.path.exists(p) and os.path.getsize(p) > 10_000:
            continue
        print(f"  téléchargement {name} ...")
        urllib.request.urlretrieve(url, p)


CHARACTERS = {"sonic": prepare_sonic, "hornet": prepare_hornet}


def main():
    wanted = sys.argv[1:] or list(CHARACTERS)
    for name in wanted:
        if name not in CHARACTERS:
            sys.exit(f"Personnage inconnu : {name} (choix : {', '.join(CHARACTERS)})")
        CHARACTERS[name]()
    download_models()


if __name__ == "__main__":
    main()
