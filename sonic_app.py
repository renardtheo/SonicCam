"""
sonic_app.py  -  Sonic Cam, version application (Windows / macOS / Linux)
========================================================================
Petit panneau de contrôle : aperçu, état, « Claque maintenant », pause, choix de la webcam,
fréquence et force des effets. Sous Windows, fermer la fenêtre laisse Sonic Cam tourner
dans la zone de notification (près de l'horloge).
"""
import base64
import os
import queue
import socket
import sys
import threading
import time

# appli sans console (PyInstaller --windowed) : stdout/stderr n'existent pas
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import tkinter as tk
from tkinter import ttk

import cv2

import sonic_cam as sc

FREQS = [("Test : 10 à 30 s", 10, 30), ("1 à 3 min", 60, 180),
         ("5 à 15 min", 300, 900), ("15 à 30 min", 900, 1800)]
EFFECTS = [("Discrets", 0.5), ("Normaux", 1.0), ("Forts", 1.4)]
AUTO = "Automatique"
PREVIEW_W = 360
SINGLE_INSTANCE_PORT = 47653


class App:
    def __init__(self, lock_sock, debug=False):
        self.lock_sock = lock_sock
        self.cfg = sc.load_config()
        self.cfg.pop("paused", None)
        self.cfg["debug"] = debug           # jamais sauvegardé : chaque lancement repart en prod
        self.log_lock = threading.Lock()
        self.logf = open(os.path.join(sc.config_dir(), "sonic_cam.log"), "a", encoding="utf-8")
        sc.LOG = self.log
        self.ui_q = queue.Queue()
        self.latest = None
        self.engine = None
        self.tray = None
        self.paused = False
        self.cams = []
        self.hidden_once = False
        self._photo = None

        self.root = tk.Tk()
        self.root.title("Sonic Cam")
        self.root.resizable(False, False)
        icon = os.path.join(sc.ASSETS, "icon.png")
        if os.path.exists(icon):
            self._icon = tk.PhotoImage(file=icon)
            self.root.iconphoto(True, self._icon)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_ui()
        self.log("Sonic Cam démarre")
        threading.Thread(target=self._load_cameras, daemon=True).start()
        self.start_engine()
        if sc.IS_WIN:
            self._start_tray()
        self.root.after(80, self.tick)

    # ─────────────────────────────────────────────── interface
    def _build_ui(self):
        f = ttk.Frame(self.root, padding=12)
        f.grid(sticky="nsew")
        blank = tk.PhotoImage(width=PREVIEW_W, height=PREVIEW_W * 9 // 16)
        self.preview = tk.Label(f, image=blank, bg="#111", width=PREVIEW_W, height=PREVIEW_W * 9 // 16)
        self.preview.image = blank
        self.preview.grid(row=0, column=0, columnspan=3, pady=(0, 8))

        self.lbl_state = ttk.Label(f, text="Démarrage…", font=("TkDefaultFont", 10, "bold"))
        self.lbl_state.grid(row=1, column=0, columnspan=3, sticky="w")
        self.lbl_info = ttk.Label(f, text="", wraplength=PREVIEW_W, foreground="#555")
        self.lbl_info.grid(row=2, column=0, columnspan=3, sticky="w")
        self.lbl_err = tk.Label(f, text="", wraplength=PREVIEW_W, fg="#c0392b", justify="left")
        self.lbl_err.grid(row=3, column=0, columnspan=3, sticky="w")

        b = ttk.Frame(f)
        b.grid(row=4, column=0, columnspan=3, pady=8, sticky="ew")
        ttk.Button(b, text="Claque maintenant", command=lambda: self.trigger("slap")).pack(side="left")
        ttk.Button(b, text="Passage", command=lambda: self.trigger("passby")).pack(side="left", padx=6)
        self.btn_pause = ttk.Button(b, text="Pause", command=self.toggle_pause)
        self.btn_pause.pack(side="left")
        self.btn_retry = ttk.Button(b, text="Réessayer", command=self.restart_engine)

        ttk.Label(f, text="Webcam").grid(row=5, column=0, sticky="w", pady=2)
        self.cmb_cam = ttk.Combobox(f, state="readonly", width=30, values=[AUTO])
        self.cmb_cam.set(self._cam_label(self.cfg.get("camera")))
        self.cmb_cam.grid(row=5, column=1, sticky="ew")
        self.cmb_cam.bind("<<ComboboxSelected>>", self.on_camera)
        ttk.Button(f, text="↻", width=3, command=lambda: threading.Thread(
            target=self._load_cameras, daemon=True).start()).grid(row=5, column=2, padx=(4, 0))

        ttk.Label(f, text="Fréquence").grid(row=6, column=0, sticky="w", pady=2)
        self.cmb_freq = ttk.Combobox(f, state="readonly", width=30, values=[x[0] for x in FREQS])
        self.cmb_freq.set(self._freq_label())
        self.cmb_freq.grid(row=6, column=1, sticky="ew")
        self.cmb_freq.bind("<<ComboboxSelected>>", self.on_freq)

        ttk.Label(f, text="Effets").grid(row=7, column=0, sticky="w", pady=2)
        self.cmb_fx = ttk.Combobox(f, state="readonly", width=30, values=[x[0] for x in EFFECTS])
        self.cmb_fx.set(min(EFFECTS, key=lambda e: abs(e[1] - self.cfg["impact"]))[0])
        self.cmb_fx.grid(row=7, column=1, sticky="ew")
        self.cmb_fx.bind("<<ComboboxSelected>>", self.on_effects)

        self.var_mirror = tk.BooleanVar(value=bool(self.cfg.get("mirror")))
        ttk.Checkbutton(f, text="Image miroir", variable=self.var_mirror,
                        command=self.on_mirror).grid(row=8, column=1, sticky="w", pady=2)

        self.var_debug = tk.BooleanVar(value=bool(self.cfg.get("debug")))
        ttk.Checkbutton(f, text="Mode debug (sans limite de fps + mesures)", variable=self.var_debug,
                        command=self.on_debug).grid(row=9, column=1, sticky="w", pady=2)
        self.lbl_debug = tk.Label(f, text="", justify="left", anchor="w", font=("TkFixedFont", 8),
                                  fg="#7d5a00", wraplength=PREVIEW_W)
        self.lbl_debug.grid(row=10, column=0, columnspan=3, sticky="w")

        self.lbl_hint = ttk.Label(f, text="", wraplength=PREVIEW_W, foreground="#2c6fbb")
        self.lbl_hint.grid(row=11, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _cam_label(self, cam):
        if cam is None:
            return AUTO
        if isinstance(cam, int) or str(cam).isdigit():
            return f"Caméra {cam}"
        return str(cam)

    def _freq_label(self):
        for name, lo, hi in FREQS:
            if lo == self.cfg["min_gap"] and hi == self.cfg["max_gap"]:
                return name
        return f"{fmt_m(self.cfg['min_gap'])} à {fmt_m(self.cfg['max_gap'])}"

    # ─────────────────────────────────────────────── moteur
    def start_engine(self):
        self.latest = None
        self.engine = sc.Engine(self.cfg, on_frame=self._on_frame)
        self.engine.set_paused(self.paused)
        self.engine.start()

    def restart_engine(self):
        old = self.engine
        self.engine = None
        self.lbl_state.config(text="Redémarrage…")

        def worker():
            if old:
                old.stop()
            self.post(self.start_engine)
        threading.Thread(target=worker, daemon=True).start()

    def _on_frame(self, img):                       # thread moteur
        self.latest = img

    def trigger(self, kind):
        if self.engine:
            self.engine.trigger(kind)

    def toggle_pause(self):
        self.paused = not self.paused
        if self.engine:
            self.engine.set_paused(self.paused)
        self.btn_pause.config(text="Reprendre" if self.paused else "Pause")
        self.log("Pause" if self.paused else "Reprise")
        if self.tray:
            self.tray.update_menu()

    # ─────────────────────────────────────────────── réglages
    def on_camera(self, _=None):
        label = self.cmb_cam.get()
        if label == AUTO:
            cam = None
        elif label.startswith("Caméra ") and label[7:].isdigit():
            cam = int(label[7:])
        else:
            cam = label
        if cam == self.cfg.get("camera"):
            return
        self.cfg["camera"] = cam
        sc.save_config(self.cfg)
        self.log(f"Webcam choisie : {label}")
        self.restart_engine()

    def on_freq(self, _=None):
        name = self.cmb_freq.get()
        for n, lo, hi in FREQS:
            if n == name:
                self.cfg["min_gap"], self.cfg["max_gap"] = float(lo), float(hi)
        sc.save_config(self.cfg)
        if self.engine:
            self.engine.set_gaps(self.cfg["min_gap"], self.cfg["max_gap"])

    def on_effects(self, _=None):
        self.cfg["impact"] = dict(EFFECTS)[self.cmb_fx.get()]
        sc.save_config(self.cfg)
        if self.engine:
            self.engine.set_impact(self.cfg["impact"])

    def on_mirror(self):
        self.cfg["mirror"] = bool(self.var_mirror.get())
        sc.save_config(self.cfg)
        if self.engine:
            self.engine.cfg["mirror"] = self.cfg["mirror"]

    def on_debug(self):
        self.cfg["debug"] = bool(self.var_debug.get())      # pas de save_config : réservé aux tests
        self.log("Mode debug activé (sans limite de fps)" if self.cfg["debug"] else "Retour en mode normal")
        self.restart_engine()                               # la cadence webcam se règle à l'ouverture

    def _load_cameras(self):                        # thread : la liste peut prendre un instant
        try:
            cams = [n for _, n in sc.list_cameras() if not sc.is_virtual(n)]
        except Exception as e:
            self.log(f"Liste des caméras impossible : {e}")
            cams = []
        self.post(lambda: self._set_cameras(cams))

    def _set_cameras(self, cams):
        values = [AUTO] + cams
        current = self._cam_label(self.cfg.get("camera"))
        if current not in values:
            values.append(current)
        self.cmb_cam.config(values=values)

    # ─────────────────────────────────────────────── boucle UI
    def post(self, fn):                             # exécute fn dans le thread Tk
        self.ui_q.put(fn)

    def tick(self):
        while True:
            try:
                self.ui_q.get_nowait()()
            except queue.Empty:
                break
        img = self.latest
        if img is not None and self.root.state() != "withdrawn":
            h = PREVIEW_W * img.shape[0] // img.shape[1]
            small = cv2.resize(img, (PREVIEW_W, h), interpolation=cv2.INTER_AREA)
            ok, png = cv2.imencode(".png", small)
            if ok:
                self._photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()).decode("ascii"), format="png")
                self.preview.config(image=self._photo, height=h)
        self._refresh_labels()
        self.root.after(80, self.tick)

    def _refresh_labels(self):
        e = self.engine
        if e is None:
            return
        st = e.status
        if st["state"] == "erreur" or (not e.alive() and st["state"] != "démarrage"):
            self.lbl_state.config(text="● Arrêté", foreground="#c0392b")
            self.btn_retry.pack(side="left", padx=6)
        elif st["state"] == "actif":
            self.btn_retry.pack_forget()
            if e.director and e.director.event is not None:
                txt = "● Sonic est là !"
            elif self.paused:
                txt = "● En pause (les boutons marchent toujours)"
            else:
                n = e.next_in()
                txt = f"● Actif, prochain passage dans {sc.fmt_delay(n)}" if n is not None else "● Actif"
            self.lbl_state.config(text=txt, foreground="#1e8449")
        else:
            self.lbl_state.config(text="Démarrage de la webcam…", foreground="#555")
        parts = []
        if st["camera"]:
            parts.append(f"Webcam : {st['camera']}")
        if st["vcam"]:
            parts.append(f"Sortie : {st['vcam']}")
        self.lbl_info.config(text="   ·   ".join(parts))
        self.lbl_err.config(text=st["error"])
        self.lbl_debug.config(text="\n".join(e.debug_lines()) if e.debug_mode and st["state"] == "actif" else "")
        self.lbl_hint.config(text=f"Dans Teams / Discord, choisissez la caméra « {st['vcam']} »."
                             if st["vcam"] else "")

    # ─────────────────────────────────────────────── zone de notification (Windows)
    def _start_tray(self):
        try:
            import pystray
            from PIL import Image
            img = Image.open(os.path.join(sc.ASSETS, "icon.png"))
            M = pystray.MenuItem
            menu = pystray.Menu(
                M("Afficher Sonic Cam", lambda: self.post(self.show), default=True),
                M("Claque maintenant", lambda: self.post(lambda: self.trigger("slap"))),
                M("Passage", lambda: self.post(lambda: self.trigger("passby"))),
                M(lambda _: "Reprendre" if self.paused else "Pause", lambda: self.post(self.toggle_pause)),
                pystray.Menu.SEPARATOR,
                M("Quitter", lambda: self.post(self.quit)),
            )
            self.tray = pystray.Icon("SonicCam", img, "Sonic Cam", menu)
            self.tray.run_detached()
        except Exception as e:
            self.tray = None
            self.log(f"Icône de notification indisponible : {e}")

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def on_close(self):
        if self.tray:
            self.root.withdraw()
            if not self.hidden_once:
                self.hidden_once = True
                try:
                    self.tray.notify("Sonic Cam continue de tourner. Clic droit sur l'icône pour quitter.",
                                     "Sonic Cam")
                except Exception:
                    pass
        else:
            self.quit()

    def quit(self):
        self.log("Fermeture")
        if self.engine:
            self.engine.stop()
        if self.tray:
            self.tray.stop()
        self.root.destroy()

    # ─────────────────────────────────────────────── journal
    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        with self.log_lock:
            try:
                self.logf.write(line + "\n")
                self.logf.flush()
            except Exception:
                pass
        try:
            print(line, flush=True)
        except Exception:
            pass

    def run(self):
        self.root.mainloop()


def fmt_m(sec):
    return f"{sec / 60:g} min" if sec >= 60 else f"{sec:g} s"


def selftest(out_path):
    """Vérifie l'appli packagée sans webcam ni fenêtre (utilisé par la CI) : écrit OK/FAIL dans out_path."""
    lines = []
    try:
        import mediapipe  # noqa: F401
        import pyvirtualcam  # noqa: F401
        if sc.IS_WIN:
            import pystray  # noqa: F401
            lines.append(f"cameras: {sc.list_cameras()}")
        sc.SPRITE = sc.Sprite()
        lines.append(f"sprite: {len(sc.SPRITE.frames)} frames")
        v = sc.Vision()
        frame = sc.SimulatedScene(640, 360).read()
        v.face(frame, 1.0)
        m = v.person_mask(frame, 1.1)
        lines.append(f"vision: mask {m.shape}")
        eng = sc.Engine(sc.DEFAULTS, simulate=True, vcam=False, first_delay=0.2, duration=6)
        eng.cfg.update(min_gap=1, max_gap=2, slap_ratio=1.0)
        eng.running = True
        eng._run()
        if eng.status["error"]:
            raise RuntimeError(eng.status["error"])
        if eng.status["fps"] < 5:
            raise RuntimeError("le moteur ne tourne pas")
        lines.append(f"engine: {eng.status['fps']:.0f} fps")
        lines.append("OK")
        code = 0
    except Exception:
        import traceback
        lines.append(traceback.format_exc())
        lines.append("FAIL")
        code = 1
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    return code


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(selftest(sys.argv[2]))
    # une seule instance : deux Sonic Cam se disputeraient la webcam et la caméra virtuelle
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", SINGLE_INSTANCE_PORT))
    except OSError:
        r = tk.Tk()
        r.withdraw()
        from tkinter import messagebox
        messagebox.showinfo("Sonic Cam", "Sonic Cam est déjà lancé (regardez près de l'horloge).")
        return
    App(lock, debug="--debug" in sys.argv).run()


if __name__ == "__main__":
    main()
