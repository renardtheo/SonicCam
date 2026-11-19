"""
sonic_cam.py  -  Caméra "Sonic Slap" pour Teams / Discord
=========================================================
Webcam -> détection visage + segmentation de la personne (MediaPipe)
       -> Sonic (sprite détouré) qui apparaît au hasard, marche devant OU derrière vous,
          saute à côté de votre tête et vous colle une claque (tête qui part, shake, étincelles, joue rouge)
          (ou Hornet, au choix, qui passe en dansant : pas de claque)
       -> caméra virtuelle OBS  (à choisir dans Teams / Discord : "OBS Virtual Camera")

Touches (fenêtre d'aperçu) :  S = claque maintenant   P = passage derrière   D = debug   Q / Echap = quitter
(S fait un simple passage si le personnage ne sait pas claquer)
Lancer d'abord : python prepare_sonic.py
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
# dans l'appli packagée (PyInstaller), les assets sont décompressés dans sys._MEIPASS
ASSETS = os.path.join(getattr(sys, "_MEIPASS", ROOT), "assets")

IS_WIN = sys.platform.startswith("win")
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
CAP_API = cv2.CAP_DSHOW if IS_WIN else (cv2.CAP_AVFOUNDATION if IS_MAC else cv2.CAP_V4L2)


def LOG(msg):
    """Journal (remplacé par l'appli pour afficher/écrire dans un fichier)."""
    print(msg, flush=True)


def fmt_delay(sec):
    sec = int(round(sec))
    return f"{sec // 60} min {sec % 60:02d} s" if sec >= 60 else f"{sec} s"


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def lerp(a, b, t):
    return a + (b - a) * t


# ════════════════════════════════════════════════════════════════════════════════
#  Entrées : webcam (thread) / fichier vidéo / scène simulée
# ════════════════════════════════════════════════════════════════════════════════
class CamReader(threading.Thread):
    """Lit la webcam en continu et garde seulement la dernière image (latence minimale)."""

    def __init__(self, source, width, height, fps):
        super().__init__(daemon=True)
        self.is_file = not isinstance(source, int) and not str(source).isdigit()
        if self.is_file:
            self.cap = cv2.VideoCapture(source)
        else:
            # Réglages passés DÈS l'ouverture : sous Windows (DirectShow), un cap.set(FOURCC) fait après coup
            # est ignoré -> la webcam reste en YUY2, souvent limité à ~10 fps en 1280x720 (contre 30 en MJPG).
            params = [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"),
                      cv2.CAP_PROP_FRAME_WIDTH, width, cv2.CAP_PROP_FRAME_HEIGHT, height, cv2.CAP_PROP_FPS, fps]
            self.cap = cv2.VideoCapture(int(source), CAP_API, params)
            if not self.cap.isOpened():                         # backend qui refuse un réglage : ancienne méthode
                self.cap = cv2.VideoCapture(int(source), CAP_API)
                for prop, val in zip(params[::2], params[1::2]):
                    self.cap.set(prop, val)
        if not self.cap.isOpened():
            raise RuntimeError(f"Impossible d'ouvrir la caméra {source}")
        self.file_dt = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30)
        ok, frame = self.cap.read()
        if not ok:
            self.cap.release()
            raise RuntimeError("La caméra ne renvoie aucune image (déjà utilisée par Teams/Discord ?)")
        self.frame = frame
        self.seq = 0                    # n° de l'image webcam (la boucle peut tourner plus vite que la caméra)
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frame
                self.seq += 1
            if self.is_file:
                time.sleep(self.file_dt)

    def info(self):
        """Format réellement négocié avec la webcam (à appeler avant start())."""
        if self.is_file:
            return f"fichier vidéo ({1 / self.file_dt:.0f} fps)"
        fc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fc >> 8 * i) & 255) for i in range(4)) if fc > 0 else "?"
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return (f"{fourcc} {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                + (f" (annonce {fps:.0f} fps)" if fps > 0 else ""))

    def read(self):
        return self.read_seq()[1]

    def read_seq(self):
        with self.lock:
            return self.seq, self.frame.copy()

    def stop(self):
        self.running = False
        if self.is_alive():
            self.join(timeout=1.0)
        self.cap.release()


class SimulatedScene:
    """Scène de test sans webcam : un mur + une silhouette qui bouge (visage et masque connus)."""

    def __init__(self, w=1280, h=720):
        self.w, self.h = w, h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        wall = np.dstack([150 + 30 * xx / w, 160 + 20 * yy / h, 175 - 10 * xx / w])
        self.wall = wall.clip(0, 255).astype(np.uint8)
        cv2.rectangle(self.wall, (int(w * .06), int(h * .12)), (int(w * .22), int(h * .45)), (90, 120, 160), -1)
        cv2.rectangle(self.wall, (int(w * .74), int(h * .18)), (int(w * .9), int(h * .4)), (120, 90, 70), -1)
        self.t0 = time.perf_counter()
        self.face = None
        self.mask = None

    def read(self):
        t = time.perf_counter() - self.t0
        w, h = self.w, self.h
        cx = w * (0.5 + 0.04 * math.sin(t * 0.7))
        cy = h * (0.40 + 0.015 * math.sin(t * 1.3))
        fw, fh = w * 0.15, h * 0.34
        m = np.zeros((h, w), np.uint8)
        cv2.ellipse(m, (int(cx), int(cy)), (int(fw * .55), int(fh * .62)), 0, 0, 360, 255, -1)
        cv2.rectangle(m, (int(cx - fw * .25), int(cy)), (int(cx + fw * .25), int(cy + fh * .9)), 255, -1)
        cv2.ellipse(m, (int(cx), int(cy + fh * 1.55)), (int(fw * 1.9), int(fh * .85)), 0, 180, 360, 255, -1)
        cv2.rectangle(m, (int(cx - fw * 1.9), int(cy + fh * 1.5)), (int(cx + fw * 1.9), h), 255, -1)
        m = cv2.GaussianBlur(m, (9, 9), 0)
        a = (m.astype(np.float32) / 255)[..., None]
        person = np.full((h, w, 3), (60, 70, 85), np.float32)
        person[: int(cy + fh * .55)] = (120, 150, 200)
        cv2.ellipse(person, (int(cx - fw * .18), int(cy - fh * .08)), (9, 6), 0, 0, 360, (30, 30, 30), -1)
        cv2.ellipse(person, (int(cx + fw * .18), int(cy - fh * .08)), (9, 6), 0, 0, 360, (30, 30, 30), -1)
        img = (self.wall * (1 - a) + person * a).astype(np.uint8)
        self.face = (cx, cy, fw, fh * 0.75)
        self.mask = m.astype(np.float32) / 255
        return img

    def stop(self):
        pass


# ════════════════════════════════════════════════════════════════════════════════
#  Vision : visage + segmentation (MediaPipe Tasks)
# ════════════════════════════════════════════════════════════════════════════════
class Vision:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mpt
        from mediapipe.tasks.python import vision as mpv
        self.mp = mp
        models = os.path.join(ASSETS, "models")
        self.face_det = mpv.FaceDetector.create_from_options(mpv.FaceDetectorOptions(
            base_options=mpt.BaseOptions(model_asset_path=os.path.join(models, "blaze_face_short_range.tflite")),
            running_mode=mpv.RunningMode.VIDEO, min_detection_confidence=0.5))
        self.seg = mpv.ImageSegmenter.create_from_options(mpv.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_path=os.path.join(models, "selfie_segmenter.tflite")),
            running_mode=mpv.RunningMode.VIDEO, output_confidence_masks=True, output_category_mask=False))
        self.ts_face = 0
        self.ts_seg = 0
        self.mask_prev = None

    def _small(self, frame):
        h, w = frame.shape[:2]
        k = 640 / w
        small = cv2.resize(frame, (640, int(h * k)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), k

    def face(self, frame, now):
        img, k = self._small(frame)
        self.ts_face = max(self.ts_face + 1, int(now * 1000))
        res = self.face_det.detect_for_video(img, self.ts_face)
        if not res.detections:
            return None
        d = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        b = d.bounding_box
        return ((b.origin_x + b.width / 2) / k, (b.origin_y + b.height / 2) / k, b.width / k, b.height / k)

    def person_mask(self, frame, now):
        img, _ = self._small(frame)
        self.ts_seg = max(self.ts_seg + 1, int(now * 1000))
        res = self.seg.segment_for_video(img, self.ts_seg)
        m = res.confidence_masks[-1].numpy_view().astype(np.float32)
        if m.ndim == 3:
            m = m[..., 0]
        if self.mask_prev is not None and self.mask_prev.shape == m.shape:
            m = 0.6 * m + 0.4 * self.mask_prev          # stabilise un peu les bords
        self.mask_prev = m
        m = cv2.GaussianBlur(m, (5, 5), 0)
        h, w = frame.shape[:2]
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip((m - 0.25) / 0.5, 0, 1)


class FaceTracker:
    """Lisse la position du visage (EMA) et garde la dernière connue."""

    def __init__(self):
        self.f = None
        self.last_seen = -1e9

    def update(self, det, now):
        if det is not None:
            if self.f is None or now - self.last_seen > 1.0:
                self.f = list(det)
            else:
                for i, a in enumerate((0.45, 0.45, 0.2, 0.2)):
                    self.f[i] += (det[i] - self.f[i]) * a
            self.last_seen = now
        return self.get(now)

    def get(self, now, max_age=2.0):
        if self.f is None or now - self.last_seen > max_age:
            return None
        return tuple(self.f)


# ════════════════════════════════════════════════════════════════════════════════
#  Sprites des personnages (assets/<perso>/meta.json + frame_XXX.png)
# ════════════════════════════════════════════════════════════════════════════════
DEFAULT_CHARACTER = "sonic"


def list_characters():
    """[(id, nom affiché)] des personnages présents dans assets/, Sonic en premier."""
    out = []
    try:
        names = sorted(os.listdir(ASSETS))
    except OSError:
        names = []
    for n in names:
        meta_p = os.path.join(ASSETS, n, "meta.json")
        if not os.path.isfile(meta_p):
            continue
        try:
            with open(meta_p, encoding="utf-8") as fp:
                label = json.load(fp).get("name") or n.capitalize()
        except Exception:
            continue
        out.append((n, label))
    out.sort(key=lambda c: (c[0] != DEFAULT_CHARACTER, c[0]))
    return out


def character_can_slap(name):
    """Le perso a-t-il une claque ? (lit seulement meta.json, sans charger les images)"""
    try:
        with open(os.path.join(ASSETS, name or DEFAULT_CHARACTER, "meta.json"), encoding="utf-8") as fp:
            meta = json.load(fp)
        return bool(meta.get("slap_seq")) and meta.get("hand") is not None
    except Exception:
        return True                                             # perso inconnu -> Sonic


class Sprite:
    def __init__(self, name=DEFAULT_CHARACTER):
        d = os.path.join(ASSETS, name)
        meta_p = os.path.join(d, "meta.json")
        if not os.path.exists(meta_p):
            raise RuntimeError(f"Assets « {name} » manquants : lancez d'abord  python prepare_sonic.py")
        with open(meta_p, encoding="utf-8") as fp:
            self.meta = json.load(fp)
        self.id = name
        self.name = self.meta.get("name") or name.capitalize()
        self.frames = [cv2.imread(os.path.join(d, f"frame_{i:03d}.png"), cv2.IMREAD_UNCHANGED)
                       for i in range(self.meta["count"])]
        self.fps = self.meta["fps"]
        self.rx, self.ry = self.meta["ref"]
        self.char_h = self.meta["char_h"]
        self.walk = self.meta["walk_loop"]
        self.walk_len = self.walk[1] - self.walk[0]
        # avance par frame d'anim (en hauteurs de perso) et vitesse de lecture du passage
        self.stride = self.meta.get("stride", STRIDE)
        self.walk_speed = tuple(self.meta.get("walk_speed", (1.5, 1.9)))
        # la claque n'existe que si la vidéo la contient (Hornet : passage seulement)
        self.seq = self.meta.get("slap_seq")
        self.can_slap = bool(self.seq) and self.meta.get("hand") is not None
        fr = self.meta["frames"]
        w0, w1 = self.walk
        self.ext_left = max(self.rx - fr[i]["ox"] for i in range(w0, w1))
        self.ext_right = max(fr[i]["ox"] + fr[i]["w"] - self.rx for i in range(w0, w1))
        if self.can_slap:
            self.impact = self.meta["impact_frame"]
            hx, hy = self.meta["hand"]
            self.hand_rel = (hx - self.rx, hy - self.ry)        # main à l'impact, relative aux pieds
            self.dash_frame = w0 + 16                           # foulée bien ouverte

    def draw(self, out, idx, ax, ay, sx, sy, flip, opacity=1.0, depth=0.0, mask=None):
        """Dessine la frame idx avec les pieds en (ax, ay). depth : 0 = devant, 1 = derrière la personne."""
        if opacity <= 0.01:
            return
        img = self.frames[idx]
        m = self.meta["frames"][idx]
        left, top = m["ox"] - self.rx, m["oy"] - self.ry
        if flip:
            img = img[:, ::-1]
            left = -(left + m["w"])
        dw, dh = max(1, int(round(m["w"] * sx))), max(1, int(round(m["h"] * sy)))
        x0, y0 = int(round(ax + left * sx)), int(round(ay + top * sy))
        H, W = out.shape[:2]
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + dw), min(H, y0 + dh)
        if X1 <= X0 or Y1 <= Y0:
            return
        interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR
        spr = cv2.resize(np.ascontiguousarray(img), (dw, dh), interpolation=interp)[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        if depth > 0.5:                                         # profondeur de champ : un poil flou derrière
            spr = cv2.GaussianBlur(spr, (3, 3), 0)
        a = spr[..., 3:4].astype(np.float32) * (opacity / 255.0)
        col = spr[..., :3].astype(np.float32)
        if depth > 0:
            col *= 1.0 - 0.12 * depth                           # un peu plus sombre au fond
            if mask is not None:
                a *= 1.0 - depth * mask[Y0:Y1, X0:X1, None]     # occlusion par la personne
        roi = out[Y0:Y1, X0:X1].astype(np.float32)
        out[Y0:Y1, X0:X1] = (roi + (col - roi) * a).astype(np.uint8)


def draw_shadow(out, cx, cy, rx, ry, strength, depth=0.0, mask=None):
    if strength <= 0.01 or rx < 2:
        return
    H, W = out.shape[:2]
    x0, x1 = int(max(0, cx - rx * 1.6)), int(min(W, cx + rx * 1.6))
    y0, y1 = int(max(0, cy - ry * 1.6)), int(min(H, cy + ry * 1.6))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    g = np.exp(-(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2) * 1.6) * strength
    if depth > 0 and mask is not None:
        g *= 1.0 - depth * mask[y0:y1, x0:x1]
    roi = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (roi * (1 - g[..., None])).astype(np.uint8)


# ════════════════════════════════════════════════════════════════════════════════
#  Chorégraphies
# ════════════════════════════════════════════════════════════════════════════════
@dataclass
class Draw:
    idx: int
    x: float
    y: float
    sx: float
    sy: float
    flip: bool
    opacity: float = 1.0
    depth: float = 0.0
    shadow: float = 0.0        # intensité de l'ombre au sol (0 = pas d'ombre)


class Ctx:
    """Ce qu'une chorégraphie peut consulter pendant une frame."""

    def __init__(self, W, H, sprite, face, get_mask, size_mult):
        self.W, self.H, self.sp, self.face = W, H, sprite, face
        self.get_mask = get_mask
        self.size_mult = size_mult

    def front_scale(self, face):
        fh = face[3] if face else self.H * 0.3
        h_px = float(np.clip(self.size_mult * fh, 0.2 * self.H, 0.55 * self.H))
        return h_px / self.sp.char_h

    def behind_ground(self, face):
        if face is None:
            return self.H * 0.72
        return min(self.H * 0.95, face[1] + 0.8 * face[3])

    def shoulder_y(self, x, face):
        """Haut de la silhouette (épaule) sous le visage, à l'abscisse x."""
        m = self.get_mask()
        if m is None or face is None:
            return None
        cx, cy, fw, fh = face
        x0, x1 = int(max(0, x - 0.1 * fw)), int(min(self.W, x + 0.1 * fw))
        y0 = int(min(self.H - 1, cy + 0.3 * fh))
        if x1 <= x0:
            return None
        rows = np.where(m[y0:, x0:x1].mean(axis=1) > 0.5)[0]
        return float(y0 + rows[0]) if len(rows) else None

    def person_edge(self, y_top, y_bot, direction):
        """Bord de la silhouette (côté direction) dans la bande verticale de Sonic."""
        m = self.get_mask()
        if m is None:
            return None
        y0, y1 = int(max(0, y_top)), int(min(self.H, y_bot))
        if y1 <= y0:
            return None
        cols = np.where(m[y0:y1].max(axis=0) > 0.5)[0]
        if len(cols) == 0:
            return None
        return float(cols.max() if direction > 0 else cols.min())


class PassBy:
    """Le perso traverse l'écran, le plus souvent DERRIÈRE la personne."""
    name = "passby"

    def __init__(self, ctx, rng):
        sp = ctx.sp
        self.dir = rng.choice((-1, 1))
        self.flip = self.dir < 0
        self.depth = 1.0 if rng.random() < 0.75 else 0.0
        self.speed = rng.uniform(*sp.walk_speed)
        s = ctx.front_scale(ctx.face) * rng.uniform(0.9, 1.1)
        if self.depth > 0:
            self.s = s * 0.72
            self.ground = ctx.behind_ground(ctx.face)
        else:
            self.s = s * 0.85
            self.ground = ctx.H * rng.uniform(0.96, 1.0)
        lead, trail = (sp.ext_right, sp.ext_left) if self.dir > 0 else (sp.ext_left, sp.ext_right)
        self.x0 = -lead * self.s - 10 if self.dir > 0 else ctx.W + lead * self.s + 10
        self.x1 = ctx.W + trail * self.s + 10 if self.dir > 0 else -trail * self.s - 10
        self.v = sp.stride * sp.char_h * self.s                 # px par frame d'anim (pieds qui ne glissent pas)
        self.phase = rng.randrange(sp.walk_len)
        self.a = 0.0
        self.done = False
        self.needs_mask = self.depth > 0

    def update(self, dt, ctx, fx):
        sp = ctx.sp
        self.a += dt * sp.fps * self.speed
        x = self.x0 + self.dir * self.v * self.a
        if (x - self.x1) * self.dir > 0:
            self.done = True
            return []
        idx = sp.walk[0] + (int(self.a) + self.phase) % sp.walk_len
        return [Draw(idx, x, self.ground, self.s, self.s, self.flip, 1.0, self.depth, 0.22)]


class SlapEvent:
    """Arrive (devant ou derrière), saute à côté de la tête, CLAQUE, puis repart en dash."""
    name = "slap"
    TURN_END = 12      # frames (dans la séquence) de décélération
    HOP_END = 34       # fin du saut vers l'épaule
    CHEEK_Y = 0.2      # hauteur de l'impact sur le visage (0 = centre, 0.5 = menton)

    def __init__(self, ctx, rng):
        sp, face, W, H = ctx.sp, ctx.face, ctx.W, ctx.H
        self.rng = rng
        self.sp = sp
        cx, cy, fw, fh = face
        self.s = ctx.front_scale(face) * rng.uniform(0.92, 1.08)

        # côté : Sonic se place à DROITE du visage quand dir=+1 (il claque vers la gauche, comme dans la vidéo)
        room_r = W - (cx + 0.4 * fw - sp.hand_rel[0] * self.s + sp.ext_right * self.s * 0.6)
        room_l = (cx - 0.4 * fw + sp.hand_rel[0] * self.s - sp.ext_right * self.s * 0.6)
        if room_r > 0 and room_l > 0:
            self.dir = rng.choice((-1, 1))
        else:
            self.dir = 1 if room_r >= room_l else -1
        self.flip = self.dir < 0
        # pieds posés sur l'épaule : Sonic grandit un peu si l'épaule est plus bas (max +35 %), sinon il "flotte"
        shoulder = ctx.shoulder_y(self.slap_anchor(face)[0], face)
        if shoulder is not None:
            need = (shoulder - (cy + self.CHEEK_Y * fh)) / -sp.hand_rel[1]
            self.s = float(np.clip(need, self.s, self.s * 1.35))
        self.last_anchor = self.slap_anchor(face)

        self.behind = rng.random() < 0.6                         # passe derrière avant de surgir devant
        self.speed = rng.uniform(1.5, 1.9)                     # marche rapide (c'est Sonic)
        self.seq_speed = rng.uniform(1.0, 1.2)                 # claque lisible
        self.hold = rng.uniform(0.15, 0.45)                     # petit regard caméra avant de filer
        if self.behind:
            self.s_walk = self.s * 0.72
            self.ground = ctx.behind_ground(face)
            self.depth0 = 1.0
        else:
            self.s_walk = self.s * 0.9
            self.ground = H * rng.uniform(0.96, 1.0)
            self.depth0 = 0.0
        self.v = sp.stride * sp.char_h * self.s_walk

        # point de départ hors champ
        self.x0 = -sp.ext_right * self.s_walk - 10 if self.dir > 0 else W + sp.ext_right * self.s_walk + 10
        # point d'arrêt : au-delà de l'épaule si on vient de derrière (pour surgir sans traverser le corps)
        ax, _ = self.slap_anchor(face)
        x_stop = ax + self.dir * 0.15 * fw
        if self.behind:
            edge = ctx.person_edge(self.ground - sp.char_h * self.s_walk, self.ground, self.dir)
            if edge is None:
                edge = cx + self.dir * 1.6 * fw
            # pieds au-delà du bord + largeur arrière du sprite -> Sonic est entièrement sorti de derrière vous
            x_stop = edge + self.dir * sp.ext_left * self.s_walk * 0.8
            x_stop = max(x_stop, ax) if self.dir > 0 else min(x_stop, ax)
        margin = sp.ext_right * self.s_walk * 0.5
        x_stop = float(np.clip(x_stop, margin, W - margin))
        # la marche se termine exactement sur une fin de cycle -> enchaînement propre avec la séquence de claque
        x_walk_end = x_stop - self.dir * self.v * self.TURN_END / 2
        dist = abs(x_walk_end - self.x0)
        n = max(36, int(round(dist / self.v / 36)) * 36)
        self.v_walk = dist / n
        self.n_walk = n
        self.x_walk_end = x_walk_end
        self.x_stop = x_stop
        self.a = 0.0
        self.t_dash = None
        self.impacted = False
        self.done = False
        self.needs_mask = self.behind

    def slap_anchor(self, face):
        """Position des pieds pour que la main tombe pile sur la joue."""
        cx, cy, fw, fh = face
        cheek_x = cx + self.dir * 0.36 * fw
        cheek_y = cy + self.CHEEK_Y * fh
        hx, hy = self.ctx_hand()
        return cheek_x - hx, cheek_y - hy

    def ctx_hand(self):
        hx, hy = self.sp.hand_rel
        return (-hx if self.flip else hx) * self.s, hy * self.s

    def update(self, dt, ctx, fx):
        sp, face = ctx.sp, ctx.face
        self.a += dt * sp.fps * (self.speed if self.a < self.n_walk else self.seq_speed)
        a = self.a
        ax, ay = self.slap_anchor(face) if face else self.last_anchor
        self.last_anchor = (ax, ay)
        ax = float(np.clip(ax, 0, ctx.W))

        # 1) marche
        if a < self.n_walk:
            idx = sp.walk[0] + int(a) % (sp.walk[1] - sp.walk[0])
            x = self.x0 + self.dir * self.v_walk * a
            return [Draw(idx, x, self.ground, self.s_walk, self.s_walk, self.flip, 1, self.depth0, 0.22)]

        k = a - self.n_walk
        seq_len = sp.seq[1] - sp.seq[0]
        # 2) séquence : arrêt -> saut sur l'épaule -> claque
        if k < seq_len:
            idx = sp.seq[0] + int(k)
            if k < self.TURN_END:
                x = self.x_walk_end + self.dir * self.v * (k - k * k / (2 * self.TURN_END))
                return [Draw(idx, x, self.ground, self.s_walk, self.s_walk, self.flip, 1, self.depth0, 0.22)]
            if k < self.HOP_END:
                u = (k - self.TURN_END) / (self.HOP_END - self.TURN_END)
                e = smoothstep(u)
                x = lerp(self.x_stop, ax, e)
                arc = 0.45 * sp.char_h * self.s
                y = lerp(self.ground, ay, e) - arc * 4 * u * (1 - u)
                s = lerp(self.s_walk, self.s, e)
                # anticipation : petit écrasement au décollage, étirement en l'air
                squash = 1 - 0.12 * math.sin(math.pi * min(u / 0.2, 1)) if u < 0.2 else 1 + 0.06 * math.sin(math.pi * u)
                depth = self.depth0 * (1 - smoothstep((u - 0.1) / 0.45))
                return [Draw(idx, x, y, s * (2 - squash) ** 0.5, s * squash, self.flip, 1, depth,
                             0.22 * (1 - e))]
            if not self.impacted and idx >= sp.impact:
                self.impacted = True
                hx, hy = self.ctx_hand()
                fx.trigger_impact(ax + hx, ay + hy, -self.dir, face)
            return [Draw(idx, ax, ay, self.s, self.s, self.flip)]

        # 3) petite pause, puis dash de sortie avec traînées
        hold_frames = self.hold * sp.fps * self.seq_speed
        if k < seq_len + hold_frames:
            return [Draw(sp.seq[1] - 1, ax, ay, self.s, self.s, self.flip)]
        if self.t_dash is None:
            self.t_dash = 0.0
            self.dash_from = (ax, ay)
        self.t_dash += dt
        T_anti, T = 0.09, 0.38
        x0, y0 = self.dash_from
        flip = self.dir < 0                                     # il file dans le sens de sa marche
        if self.t_dash < T_anti:                                # anticipation (écrasement)
            p = self.t_dash / T_anti
            sq = 1 - 0.15 * math.sin(math.pi * p * 0.5)
            return [Draw(sp.dash_frame, x0, y0, self.s * (2 - sq), self.s * sq, flip)]
        p = (self.t_dash - T_anti) / T
        if p >= 1:
            self.done = True
            return []
        draws = []
        dist = ctx.W * 1.3
        for g in (3, 2, 1, 0):                                  # traînées fantômes puis Sonic
            pg = max(0.0, p - 0.07 * g)
            x = x0 + self.dir * dist * pg * pg
            y = y0 - 0.15 * sp.char_h * self.s * math.sin(math.pi * min(pg * 1.6, 1))
            st = 1 + 0.9 * pg
            draws.append(Draw(sp.dash_frame, x, y, self.s * st, self.s / st ** 0.3, flip,
                              1.0 if g == 0 else 0.4 / g))
        return draws


STRIDE = 0.021      # avance par frame d'anim, en hauteurs de Sonic (calé sur la foulée de la vidéo)
_SPRITES = {}       # cache {id: Sprite} : chaque perso n'est chargé qu'une fois
_SPRITES_LOCK = threading.Lock()


def get_sprite(name=None):
    """Sprite du perso demandé (chargé une fois). Perso inconnu -> Sonic."""
    name = name or DEFAULT_CHARACTER
    with _SPRITES_LOCK:
        if name not in _SPRITES:
            if os.path.isfile(os.path.join(ASSETS, name, "meta.json")) or name == DEFAULT_CHARACTER:
                _SPRITES[name] = Sprite(name)
            else:                                               # mémorisé : prévenu une seule fois
                LOG(f"Personnage « {name} » introuvable, retour à {DEFAULT_CHARACTER}")
                if DEFAULT_CHARACTER not in _SPRITES:
                    _SPRITES[DEFAULT_CHARACTER] = Sprite(DEFAULT_CHARACTER)
                _SPRITES[name] = _SPRITES[DEFAULT_CHARACTER]
        return _SPRITES[name]


# ════════════════════════════════════════════════════════════════════════════════
#  Effets d'impact
# ════════════════════════════════════════════════════════════════════════════════
class ImpactFX:
    def __init__(self, strength=1.0):
        self.t0 = None
        self.strength = strength

    def trigger_impact(self, hx, hy, slap_dir, face):
        self.t0 = time.perf_counter()
        self.hit = (hx, hy)
        self.slap_dir = slap_dir
        self.side = -slap_dir                                   # côté du visage frappé
        self.face0 = face
        self.rays = [random.uniform(-1, 1) for _ in range(7)]

    def active(self, now):
        return self.t0 is not None and now - self.t0 < 3.5

    def jolt(self, tau, fw):
        """Déplacement horizontal de la tête : ressort amorti."""
        if tau > 0.9:
            return 0.0
        return self.slap_dir * 0.28 * fw * self.strength * math.exp(-tau / 0.17) * math.sin(2 * math.pi * tau / 0.36 + 0.25)

    def warp_head(self, img, now, face):
        if self.t0 is None or face is None:
            return img
        tau = now - self.t0
        dx = self.jolt(tau, face[2])
        if abs(dx) < 0.5:
            return img
        cx, cy, fw, fh = face
        H, W = img.shape[:2]
        x0, x1 = int(max(0, cx - 2.6 * fw)), int(min(W, cx + 2.6 * fw))
        y0, y1 = int(max(0, cy - 1.8 * fh)), int(min(H, cy + 2.2 * fh))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        sy = np.where(yy > cy, 0.75 * fh, 1.2 * fh)             # le cou bouge moins que le crâne
        wgt = np.exp(-0.5 * (((xx - cx) / (0.95 * fw)) ** 2 + ((yy - cy) / sy) ** 2))
        shift = dx * wgt * (1 + 0.35 * (cy - yy) / fh)          # le haut part plus -> inclinaison
        out = img.copy()
        mx = (xx - shift).astype(np.float32)
        my = (yy + 0.12 * np.abs(shift)).astype(np.float32)
        out[y0:y1, x0:x1] = cv2.remap(img, mx, my, cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT)
        return out

    def overlay(self, img, now, face):
        if self.t0 is None:
            return
        tau = now - self.t0
        # joue rouge (suit le visage)
        if face is not None and tau < 3.5:
            cx, cy, fw, fh = face
            a = 0.38 * self.strength * (1 if tau < 1.2 else max(0, 1 - (tau - 1.2) / 2.3))
            bx = cx + self.side * 0.27 * fw + self.jolt(tau, fw) * 0.9
            by = cy + 0.18 * fh
            rx, ry = 0.17 * fw, 0.12 * fh
            H, W = img.shape[:2]
            x0, x1 = int(max(0, bx - 2 * rx)), int(min(W, bx + 2 * rx))
            y0, y1 = int(max(0, by - 2 * ry)), int(min(H, by + 2 * ry))
            if x1 > x0 and y1 > y0 and a > 0:
                yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
                g = (np.exp(-1.3 * (((xx - bx) / rx) ** 2 + ((yy - by) / ry) ** 2)) * a)[..., None]
                roi = img[y0:y1, x0:x1].astype(np.float32)
                red = np.array([60, 60, 235], np.float32)
                img[y0:y1, x0:x1] = (roi * (1 - g) + red * g).astype(np.uint8)
        # étincelles comiques à l'impact
        if tau < 0.16:
            p = tau / 0.16
            hx, hy = self.hit
            r_in = (8 + 30 * p) * (face[2] / 200 if face else 1)
            r_out = r_in + (28 * (1 - p) + 10) * (face[2] / 200 if face else 1)
            ov = img.copy()
            base = math.pi if self.slap_dir > 0 else 0.0        # rayons du côté opposé à la main
            for i, j in enumerate(self.rays):
                ang = base + (i - 3) * 0.42 + j * 0.12
                p1 = (int(hx + math.cos(ang) * r_in), int(hy + math.sin(ang) * r_in))
                p2 = (int(hx + math.cos(ang) * r_out), int(hy + math.sin(ang) * r_out))
                cv2.line(ov, p1, p2, (40, 40, 40), 6, cv2.LINE_AA)
                cv2.line(ov, p1, p2, (210, 250, 255), 3, cv2.LINE_AA)
            cv2.addWeighted(ov, 0.85 * (1 - p * 0.6), img, 1 - 0.85 * (1 - p * 0.6), 0, img)

    def camera_punch(self, img, now):
        """Shake + léger zoom 'punch-in' juste après l'impact."""
        if self.t0 is None:
            return img
        tau = now - self.t0
        if tau > 0.35:
            return img
        H, W = img.shape[:2]
        k = math.exp(-tau / 0.09) * self.strength
        zoom = 1 + 0.03 * k
        ang = random.uniform(0, 2 * math.pi)
        tx, ty = math.cos(ang) * 0.009 * W * k, math.sin(ang) * 0.009 * W * k
        M = np.float32([[zoom, 0, (1 - zoom) * W / 2 + tx], [0, zoom, (1 - zoom) * H / 2 + ty]])
        return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


# ════════════════════════════════════════════════════════════════════════════════
#  Réalisateur : déclenchements aléatoires
# ════════════════════════════════════════════════════════════════════════════════
class Director:
    def __init__(self, min_gap, max_gap, slap_ratio, first_delay=None):
        self.rng = random.Random()
        self.min_gap, self.max_gap = min_gap, max_gap
        self.slap_ratio = slap_ratio
        if first_delay is None:                                 # 1re apparition aussi aléatoire
            first_delay = self.rng.uniform(min_gap, max_gap)
        self.next_t = time.perf_counter() + first_delay
        self.event = None
        self.last_kind = None
        self.forced = None
        self.paused = False

    def force(self, kind):
        self.forced = kind

    def reschedule(self):
        self.next_t = time.perf_counter() + self.rng.uniform(self.min_gap, self.max_gap)

    def set_paused(self, paused):
        if self.paused and not paused:
            self.reschedule()
        self.paused = paused

    def next_in(self):
        if self.paused:
            return None
        return max(0.0, self.next_t - time.perf_counter())

    def _pick(self, sprite):
        if not sprite.can_slap:                                 # ex. Hornet : uniquement des passages
            return "passby"
        p = self.slap_ratio
        if self.last_kind == "slap":                            # évite deux claques d'affilée trop souvent
            p *= 0.6
        return "slap" if self.rng.random() < p else "passby"

    def update(self, now, dt, ctx, fx):
        if self.event is None and (self.forced or (now >= self.next_t and not self.paused)):
            kind = self.forced or self._pick(ctx.sp)
            self.forced = None
            if kind == "slap" and not ctx.sp.can_slap:
                kind = "passby"
            if kind == "slap" and ctx.face is None:             # pas de visage -> on retente plus tard
                self.next_t = now + 3
                return []
            self.event = SlapEvent(ctx, self.rng) if kind == "slap" else PassBy(ctx, self.rng)
            self.last_kind = kind
            e = self.event
            where = "derrière" if getattr(e, "behind", getattr(e, "depth", 0) > 0) else "devant"
            LOG(f"{ctx.sp.name} : {kind} ({where}, {'gauche->droite' if e.dir > 0 else 'droite->gauche'})")
        if self.event is None:
            return []
        draws = self.event.update(dt, ctx, fx)
        if self.event.done:
            self.event = None
            self.next_t = now + self.rng.uniform(self.min_gap, self.max_gap)
            LOG(f"Prochain passage dans {fmt_delay(self.next_t - now)}")
        return draws




# ════════════════════════════════════════════════════════════════════════════════
#  Caméras réelles (Windows / macOS / Linux)
# ════════════════════════════════════════════════════════════════════════════════
VIRTUAL_HINTS = ("obs virtual", "virtual cam", "unity video capture", "xsplit", "snap camera",
                 "dummy video device", "v4l2loopback", "sonic cam")


def is_virtual(name):
    return any(h in name.lower() for h in VIRTUAL_HINTS)


def list_cameras():
    """[(index OpenCV, nom)] des caméras du système (virtuelles comprises)."""
    if IS_WIN:
        try:
            from pygrabber.dshow_graph import FilterGraph
            return list(enumerate(FilterGraph().get_input_devices()))
        except Exception:
            pass
    elif IS_LINUX:
        import glob
        cams = []
        for p in glob.glob("/sys/class/video4linux/video*"):
            try:
                idx = int(p.rsplit("video", 1)[1])
                if os.path.exists(os.path.join(p, "index")):     # ignore les nœuds "metadata"
                    with open(os.path.join(p, "index")) as fp:
                        if fp.read().strip() != "0":
                            continue
                with open(os.path.join(p, "name")) as fp:
                    cams.append((idx, fp.read().strip()))
            except Exception:
                continue
        if cams:
            return sorted(cams)
    # macOS (pas de noms fiables via OpenCV) ou repli
    return [(i, f"Caméra {i}") for i in range(3)]


def camera_formats(index):
    """Formats annoncés par la webcam (Windows seulement) : {(type, largeur, hauteur): fps max}."""
    if not IS_WIN:
        return {}
    try:
        from pygrabber.dshow_graph import FilterGraph
        g = FilterGraph()
        g.add_video_input_device(index)
        out = {}
        for f in g.get_input_device().get_formats():
            k = (f["media_type_str"], f["width"], f["height"])
            out[k] = max(out.get(k, 0), round(f.get("max_framerate", 0)))
        return out
    except Exception:
        return {}


class Perf:
    """Mesures du mode debug (fenêtre glissante d'1 s) : cadences et temps de chaque étape."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.loops = self.cams = 0
        self.acc = {}
        self.snap = {"out": 0.0, "cam": 0.0, "ms": {}}

    def add(self, key, t_start):
        ms = (time.perf_counter() - t_start) * 1000
        s, m = self.acc.get(key, (0.0, 0.0))
        self.acc[key] = (s + ms, max(m, ms))

    def tick(self, new_cam, now):
        """Compte une itération ; renvoie True quand une nouvelle mesure (1 s) est prête."""
        self.loops += 1
        self.cams += new_cam
        el = now - self.t0
        if el < 1.0:
            return False
        self.snap = {"out": self.loops / el, "cam": self.cams / el,
                     "ms": {k: (s / self.loops, m) for k, (s, m) in self.acc.items()}}
        self.t0, self.loops, self.cams, self.acc = now, 0, 0, {}
        return True

    def lines(self, fps_limit):
        """[cadences, temps par étape] prêts à afficher."""
        s = self.snap
        lim = "aucune" if not fps_limit else f"{fps_limit}"
        ms = "  ".join(f"{k} {a:.1f}ms (max {m:.0f})" for k, (a, m) in s["ms"].items())
        return [f"sortie {s['out']:.1f} fps (limite : {lim}) | webcam {s['cam']:.1f} fps", ms]


def probe_camera(index):
    """(ok, luminosité moyenne) : une image ~0 = cache fermé / caméra désactivée."""
    cap = cv2.VideoCapture(index, CAP_API)
    if not cap.isOpened():
        return False, 0.0
    frame = None
    for _ in range(8):                                          # les 1res images sont souvent noires
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    return frame is not None, float(frame.mean()) if frame is not None else 0.0


def resolve_camera(wanted):
    """wanted : index, morceau de nom, chemin de vidéo ou None (= auto). -> (source, nom)"""
    if wanted is not None and not isinstance(wanted, int) and os.path.exists(str(wanted)):
        return wanted, os.path.basename(str(wanted))
    cams = list_cameras()
    if wanted is not None and str(wanted).isdigit():
        i = int(wanted)
        return i, dict(cams).get(i, f"Caméra {i}")
    if wanted:
        for i, n in cams:
            if str(wanted).lower() in n.lower():
                return i, n
        LOG(f"Caméra « {wanted} » introuvable, recherche automatique...")
    fallback = None                                             # auto : 1re vraie caméra avec une image
    for i, n in cams:
        if is_virtual(n):
            continue
        ok, lum = probe_camera(i)
        if ok and lum > 8:
            return i, n
        if ok and fallback is None:
            fallback = (i, n)
    if fallback:
        return fallback
    raise RuntimeError("Aucune webcam utilisable trouvée (déjà utilisée par Teams/Discord ?)")


def vcam_help():
    if IS_WIN:
        return ("Installez OBS Studio (obsproject.com). Si OBS est ouvert avec « Démarrer la caméra virtuelle » "
                "actif, arrêtez-la : une seule appli peut l'alimenter.")
    if IS_MAC:
        return ("Installez OBS Studio (obsproject.com), cliquez une fois sur « Démarrer la caméra virtuelle » "
                "(pour installer l'extension système), arrêtez-la, puis fermez OBS.")
    return ("Activez v4l2loopback :  sudo modprobe v4l2loopback devices=1 exclusive_caps=1 "
            "card_label=\"Sonic Cam\"")


# ════════════════════════════════════════════════════════════════════════════════
#  Réglages utilisateur (un fichier par utilisateur, partagé CLI / appli)
# ════════════════════════════════════════════════════════════════════════════════
DEFAULTS = {
    "camera": None,          # None = auto
    "character": DEFAULT_CHARACTER,   # dossier dans assets/ : "sonic" (claque + passage) ou "hornet" (passage)
    "min_gap": 300.0,        # 5 min
    "max_gap": 900.0,        # 15 min
    "slap_ratio": 0.65,
    "size": 1.2,
    "impact": 1.0,
    "mirror": False,
    "width": 1280, "height": 720,
    "fps": 60,               # cadence de sortie (caméra virtuelle, animation de Sonic)
    "cam_fps": 30,           # cadence demandée à la webcam (30 max pour la plupart)
}
# Mode debug (jamais sauvegardé : on revient en prod au prochain lancement) :
#   sortie sans limite de fps, webcam à son max, mesures à l'écran et dans le journal.
DEBUG_FPS = 0                # 0 = illimité
DEBUG_CAM_FPS = 120          # la webcam répond avec son vrai maximum si elle ne peut pas


def config_dir():
    if IS_WIN:
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif IS_MAC:
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    d = os.path.join(base, "SonicCam")
    os.makedirs(d, exist_ok=True)
    return d


CONFIG = os.path.join(config_dir(), "config.json")


def load_config():
    cfg = dict(DEFAULTS)
    for p in (os.path.join(ROOT, "config.json"), CONFIG):     # ancien emplacement puis le bon
        try:
            with open(p, encoding="utf-8") as fp:
                saved = json.load(fp)
        except Exception:
            continue
        # anciennes versions : "fps": 30 était écrit d'office (pas un choix de l'utilisateur) -> nouveau défaut
        if "cam_fps" not in saved and saved.get("fps") == 30:
            saved.pop("fps")
        cfg.update(saved)
    return cfg


def save_config(cfg):
    keep = {k: cfg.get(k, v) for k, v in DEFAULTS.items()}
    with open(CONFIG, "w", encoding="utf-8") as fp:
        json.dump(keep, fp, indent=1, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════════════════════
#  Moteur : capture -> vision -> chorégraphie -> composition -> caméra virtuelle
# ════════════════════════════════════════════════════════════════════════════════
class Engine:
    def __init__(self, cfg, on_frame=None, simulate=False, vcam=True, record=None, seed=None,
                 first_delay=None, duration=0):
        self.cfg = dict(cfg)
        self.on_frame = on_frame            # callback(image BGR) pour l'aperçu
        self.simulate, self.use_vcam, self.record = simulate, vcam, record
        self.seed, self.first_delay, self.duration = seed, first_delay, duration
        self.debug_mode = bool(self.cfg.get("debug"))      # mode debug : sans limite + mesures
        self.debug = self.debug_mode                        # surimpression dans l'aperçu (touche D en CLI)
        self.out_fps = self.cfg.get("debug_fps", DEBUG_FPS) if self.debug_mode else self.cfg["fps"]
        self.cam_fps = self.cfg.get("debug_cam_fps", DEBUG_CAM_FPS) if self.debug_mode else self.cfg["cam_fps"]
        self.perf = Perf()
        self.cam_info = ""
        self.running = False
        self.thread = None
        self.director = None
        self.fx = None
        self.status = {"state": "démarrage", "camera": "", "vcam": "", "error": "", "fps": 0.0}

    def debug_lines(self):
        """Lignes de mesures à afficher par l'interface (mode debug)."""
        perf = self.perf.lines(self.out_fps)
        d = self.director
        ev = (d.event.name if d.event else f"prochain dans {fmt_delay(d.next_in() or 0)}") if d else "-"
        return [("DEBUG | " if self.debug_mode else "") + perf[0],
                f"webcam : {self.cam_info or '?'}",
                f"évènement : {ev}"] + [p.strip() for p in perf[1].split("  ") if p.strip()]

    # -- contrôle (appelable depuis un autre thread)
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread is not threading.current_thread():
            self.thread.join(timeout=3)

    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def trigger(self, kind):
        if self.director:
            self.director.force(kind)

    def set_paused(self, paused):
        self.cfg["paused"] = paused
        if self.director:
            self.director.set_paused(paused)

    def set_gaps(self, lo, hi):
        self.cfg["min_gap"], self.cfg["max_gap"] = lo, hi
        if self.director:
            self.director.min_gap, self.director.max_gap = lo, hi
            self.director.reschedule()

    def set_impact(self, v):
        self.cfg["impact"] = v
        if self.fx:
            self.fx.strength = v

    def set_character(self, name):
        """Change de perso à chaud : pris en compte dès que l'apparition en cours est finie."""
        get_sprite(name)                                        # charge les images ici, pas dans la boucle vidéo
        self.cfg["character"] = name

    def sprite(self):
        """Perso actuellement choisi (celui qui apparaîtra la prochaine fois)."""
        return get_sprite(self.cfg.get("character"))

    def next_in(self):
        return self.director.next_in() if self.director else None

    # -- boucle
    def _run(self):
        try:
            self._loop()
        except Exception as e:                                  # l'appli affiche l'erreur au lieu de planter
            self.status["error"] = str(e)
            self.status["state"] = "erreur"
            LOG(f"[erreur] {e}")
        finally:
            self.running = False

    def _loop(self):
        cfg = self.cfg
        sprite = get_sprite(cfg.get("character"))
        LOG(f"Personnage : {sprite.name}" + ("" if sprite.can_slap else " (passages uniquement)"))
        if self.simulate:
            reader, vision = SimulatedScene(cfg["width"], cfg["height"]), None
            self.status["camera"] = "simulation"
            self.cam_info = "scène simulée"
        else:
            source, name = resolve_camera(cfg.get("camera"))
            if isinstance(source, int) and is_virtual(name):
                raise RuntimeError(f"« {name} » est une caméra virtuelle : choisissez votre vraie webcam.")
            self.status["camera"] = name
            LOG(f"Webcam utilisée : [{source}] {name}")
            if self.debug_mode:
                LOG(f"MODE DEBUG : sortie {'illimitée' if not self.out_fps else f'{self.out_fps} fps'}, "
                    f"webcam demandée à {self.cam_fps} fps")
                fmts = camera_formats(source) if isinstance(source, int) else {}
                if fmts:
                    LOG("Formats annoncés par la webcam : " + ", ".join(
                        f"{t} {w}x{h}@{f}" for (t, w, h), f in
                        sorted(fmts.items(), key=lambda kv: (kv[0][0], -kv[0][1] * kv[0][2]))))
                    LOG(f"Maximum matériel de cette webcam : {max(fmts.values())} fps")
            reader = CamReader(source, cfg["width"], cfg["height"], self.cam_fps)
            self.cam_info = reader.info()
            LOG(f"Format webcam négocié : {self.cam_info}")
            reader.start()
            vision = Vision()
        vcam = writer = None
        try:
            first = reader.read()
            H, W = first.shape[:2]
            if not self.simulate and float(first.mean()) < 3:
                time.sleep(1.0)
                if float(reader.read().mean()) < 3:
                    self.status["error"] = "La webcam renvoie une image NOIRE (cache fermé ou mauvaise caméra)."
                    LOG(self.status["error"])
            if self.use_vcam:
                import pyvirtualcam
                try:
                    vcam = pyvirtualcam.Camera(W, H, self.out_fps or 60, fmt=pyvirtualcam.PixelFormat.BGR)
                    self.status["vcam"] = vcam.device
                    LOG(f"Caméra virtuelle : {vcam.device}")
                except Exception as e:
                    self.status["vcam"] = ""
                    self.status["error"] = f"Caméra virtuelle indisponible ({e}). {vcam_help()}"
                    LOG(self.status["error"])
            if self.record:
                if not self.out_fps:
                    LOG("[!] Enregistrement sans limite de fps : la vidéo est notée 60 fps, sa vitesse sera fausse.")
                writer = cv2.VideoWriter(self.record, cv2.VideoWriter_fourcc(*"mp4v"), self.out_fps or 60, (W, H))

            tracker = FaceTracker()
            self.fx = fx = ImpactFX(cfg["impact"])
            director = Director(cfg["min_gap"], cfg["max_gap"], cfg["slap_ratio"], self.first_delay)
            if self.seed is not None:
                director.rng.seed(self.seed)
                random.seed(self.seed)
            director.set_paused(bool(cfg.get("paused")))
            self.director = director
            nxt = director.next_in()
            LOG(f"Première apparition dans {fmt_delay(nxt)}" if nxt is not None else "En pause")
            self.status["state"] = "actif"
            t_start = last = last_log = time.perf_counter()
            frame_dt = 1.0 / self.out_fps if self.out_fps else 0.0     # 0 = pas de limite
            perf = self.perf = Perf()
            n_loops = n_cam = 0
            last_seq = -1
            mask_cache = {}             # {"seq": n° d'image webcam, "mask": masque} : 1 détourage par image webcam

            while self.running:
                if self.simulate:
                    seq, frame = n_loops, reader.read()
                else:
                    seq, frame = reader.read_seq()
                new_cam = seq != last_seq
                last_seq = seq
                n_loops += 1
                n_cam += new_cam
                if cfg.get("mirror"):
                    frame = cv2.flip(frame, 1)
                now = time.perf_counter()
                dt = min(now - last, 0.1)
                last = now
                self.status["fps"] = self.status["fps"] * 0.9 + (1 / max(dt, 1e-3)) * 0.1
                if perf.tick(new_cam, now) and self.debug_mode and now - last_log >= 5:
                    last_log = now
                    LOG("[debug] " + " | ".join(perf.lines(self.out_fps)))

                # vision : seulement quand la webcam a livré une nouvelle image (sortie 60 fps, webcam 30 fps)
                t0 = time.perf_counter()
                if self.simulate:
                    face = tracker.update(reader.face, now)
                    mask_cache = {"seq": seq, "mask": reader.mask}
                elif new_cam:
                    face = tracker.update(vision.face(frame, now), now)
                else:
                    face = tracker.get(now)
                perf.add("visage", t0)

                def get_mask():
                    if mask_cache.get("seq") != seq:
                        t_m = time.perf_counter()
                        mask_cache["seq"], mask_cache["mask"] = seq, vision.person_mask(frame, now)
                        perf.add("detourage", t_m)
                    return mask_cache["mask"]

                t0 = time.perf_counter()
                if director.event is None:          # changement de perso : jamais au milieu d'une apparition
                    sprite = get_sprite(cfg.get("character"))
                ctx = Ctx(W, H, sprite, face, get_mask, cfg["size"])
                draws = director.update(now, dt, ctx, fx)

                out = fx.warp_head(frame, now, face) if fx.active(now) else frame
                mask = get_mask() if any(d.depth > 0 for d in draws) else None
                for d in draws:
                    if d.shadow > 0:
                        w_sh = 0.22 * sprite.char_h * d.sx
                        draw_shadow(out, d.x, d.y, w_sh, w_sh * 0.22, d.shadow, d.depth, mask)
                for d in draws:
                    sprite.draw(out, d.idx, d.x, d.y, d.sx, d.sy, d.flip, d.opacity, d.depth, mask)
                if fx.active(now):
                    fx.overlay(out, now, face)
                    out = fx.camera_punch(out, now)
                perf.add("compo", t0)       # inclut le détourage s'il a été calculé ici

                t0 = time.perf_counter()
                if vcam:
                    vcam.send(out)
                if writer:
                    writer.write(out)
                perf.add("envoi", t0)
                if self.on_frame:
                    if self.debug:              # uniquement dans l'aperçu, jamais dans Teams / Discord
                        out = out.copy()
                        if face:
                            cx, cy, fw, fh = face
                            cv2.rectangle(out, (int(cx - fw / 2), int(cy - fh / 2)),
                                          (int(cx + fw / 2), int(cy + fh / 2)), (0, 255, 0), 2)
                        if mask is not None:
                            out[..., 1] = np.maximum(out[..., 1], (mask * 120).astype(np.uint8))
                    self.on_frame(out)

                if self.duration and now - t_start > self.duration:
                    break
                if not self.out_fps:
                    time.sleep(0)               # illimité : on laisse juste respirer l'interface
                elif vcam:
                    vcam.sleep_until_next_frame()
                else:
                    time.sleep(max(0.0, frame_dt - (time.perf_counter() - now)))
        finally:
            el = max(time.perf_counter() - t_start, 1e-3) if "t_start" in locals() else 0
            if el:
                LOG(f"Bilan : sortie {n_loops / el:.1f} fps, webcam {n_cam / el:.1f} fps")
            if not self.status["error"] or self.status["state"] == "actif":
                self.status["state"] = "arrêté"
            reader.stop()
            if vcam:
                vcam.close()
            if writer:
                writer.release()


# ════════════════════════════════════════════════════════════════════════════════
#  Ligne de commande (l'appli avec fenêtre est sonic_app.py)
# ════════════════════════════════════════════════════════════════════════════════
def choose_camera_interactive():
    cams = list_cameras()
    print("\nCaméras détectées :")
    for i, n in cams:
        if is_virtual(n):
            print(f"  [{i}] {n}   (virtuelle, ignorée)")
            continue
        ok, lum = probe_camera(i)
        etat = "IMAGE OK" if ok and lum > 8 else ("image NOIRE (cache fermé ?)" if ok else "indisponible")
        print(f"  [{i}] {n}   -> {etat}")
    choix = input("\nNuméro de la caméra à utiliser : ").strip()
    names = dict(cams)
    if not choix.isdigit() or int(choix) not in names:
        sys.exit("Choix invalide.")
    cfg = load_config()
    name = names[int(choix)]
    cfg["camera"] = int(choix) if name.startswith("Caméra ") else name
    save_config(cfg)
    print(f"OK : « {name} » mémorisée ({CONFIG})")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Caméra Sonic Slap pour Teams / Discord (ligne de commande)")
    ap.add_argument("--camera", help="numéro ou nom (même partiel) de la webcam, ou chemin d'une vidéo")
    ap.add_argument("--character", help="personnage : sonic (claque + passage) ou hornet (passage seulement)")
    ap.add_argument("--list-cameras", action="store_true", help="liste les caméras et quitte")
    ap.add_argument("--choose-camera", action="store_true", help="choisir et mémoriser la caméra")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--fps", type=int, help="cadence de sortie (défaut 60 ; 0 = illimitée)")
    ap.add_argument("--cam-fps", type=int, help="cadence demandée à la webcam (défaut 30)")
    ap.add_argument("--debug", action="store_true",
                    help="mode test : aucune limite de fps, webcam au max, mesures dans l'aperçu et la console")
    ap.add_argument("--min-gap", type=float, help="délai min entre deux apparitions (s), défaut 300")
    ap.add_argument("--max-gap", type=float, help="délai max entre deux apparitions (s), défaut 900")
    ap.add_argument("--first", type=float, help="délai avant la 1re apparition (s), défaut : aléatoire")
    ap.add_argument("--slap-ratio", type=float, help="proba claque vs simple passage")
    ap.add_argument("--size", type=float, help="taille de Sonic en hauteurs de visage")
    ap.add_argument("--impact", type=float, help="force des effets d'impact (0..1.5)")
    ap.add_argument("--mirror", action="store_true", default=None, help="effet miroir sur la webcam")
    ap.add_argument("--no-vcam", action="store_true", help="pas de caméra virtuelle (aperçu seul)")
    ap.add_argument("--no-preview", action="store_true", help="pas de fenêtre d'aperçu")
    ap.add_argument("--record", help="enregistre la sortie dans ce fichier .mp4")
    ap.add_argument("--duration", type=float, default=0, help="s'arrête après N secondes (0 = jamais)")
    ap.add_argument("--simulate", action="store_true", help="scène de test sans webcam")
    ap.add_argument("--seed", type=int, help="graine aléatoire (tests reproductibles)")
    args = ap.parse_args()

    if args.list_cameras:
        for i, n in list_cameras():
            print(f"  [{i}] {n}{'   (virtuelle)' if is_virtual(n) else ''}")
        return
    if args.choose_camera:
        choose_camera_interactive()
        return
    for k in ("camera", "character", "width", "height", "fps", "cam_fps", "min_gap", "max_gap", "slap_ratio", "size",
              "impact", "mirror"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.debug:                      # --fps / --cam-fps restent prioritaires en debug
        cfg["debug"] = True
        if args.fps is not None:
            cfg["debug_fps"] = args.fps
        if args.cam_fps is not None:
            cfg["debug_cam_fps"] = args.cam_fps

    latest = [None]
    engine = Engine(cfg, on_frame=None if args.no_preview else (lambda f: latest.__setitem__(0, f)),
                    simulate=args.simulate, vcam=not args.no_vcam, record=args.record, seed=args.seed,
                    first_delay=args.first, duration=args.duration)
    engine.start()
    win = ("[DEBUG] " if args.debug else "") + "Sonic Cam  [S] claque  [P] passage  [D] debug  [Q] quitter"
    try:
        if args.no_preview:
            while engine.alive():
                time.sleep(0.2)
            return
        while engine.alive():
            f = latest[0]
            if f is not None:
                pv = cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])))
                if engine.debug:
                    lines = engine.debug_lines()
                    cv2.rectangle(pv, (0, 0), (640, 8 + 18 * len(lines)), (0, 0, 0), -1)
                    for i, txt in enumerate(lines):
                        txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()  # OpenCV : ASCII
                        cv2.putText(pv, txt, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(win, pv)
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                engine.trigger("slap")
            elif key == ord("p"):
                engine.trigger("passby")
            elif key == ord("d"):
                engine.debug = not engine.debug
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        cv2.destroyAllWindows()
        if engine.status["error"]:
            print("[!]", engine.status["error"])


if __name__ == "__main__":
    main()
