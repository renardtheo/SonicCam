"""
build.py  -  construit l'appli autonome Sonic Cam pour l'OS COURANT
==================================================================
  Windows : dist/SonicCam/SonicCam.exe       -> dist/SonicCam-windows-x64.zip
  macOS   : dist/SonicCam.app                -> dist/SonicCam-macos-<arch>.zip
  Linux   : dist/SonicCam/SonicCam           -> dist/SonicCam-linux-x64.tar.gz

PyInstaller ne fait pas de compilation croisée : lancez ce script sur chaque OS,
ou utilisez le workflow GitHub Actions (.github/workflows/build.yml) qui fait les trois.

Usage :  python build.py
"""
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tarfile

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(ROOT, "assets")
BUILD = os.path.join(ROOT, "build")
DIST = os.path.join(ROOT, "dist")
NAME = "SonicCam"
IS_WIN, IS_MAC = sys.platform.startswith("win"), sys.platform == "darwin"


def ensure_assets():
    if os.path.exists(os.path.join(ASSETS, "sonic", "meta.json")) and \
            os.path.exists(os.path.join(ASSETS, "models", "selfie_segmenter.tflite")):
        return
    print("Assets absents -> prepare_sonic.py")
    subprocess.check_call([sys.executable, os.path.join(ROOT, "prepare_sonic.py")])


def make_icon():
    """Icône = tête de Sonic de face, tirée de l'animation."""
    import json
    from PIL import Image
    png = os.path.join(ASSETS, "icon.png")
    meta = json.load(open(os.path.join(ASSETS, "sonic", "meta.json"), encoding="utf-8"))
    im = Image.open(os.path.join(ASSETS, "sonic", "frame_219.png")).convert("RGBA")
    # tête ≈ 45 % supérieurs du perso
    head_h = int(meta["char_h"] * 0.42)
    head = im.crop((0, 0, im.width, min(im.height, head_h)))
    bbox = head.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()
    head = head.crop(bbox)
    side = max(head.size)
    sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    sq.paste(head, ((side - head.width) // 2, (side - head.height) // 2))
    sq = sq.resize((256, 256), Image.LANCZOS)
    sq.save(png)
    os.makedirs(BUILD, exist_ok=True)
    ico = os.path.join(BUILD, "icon.ico")
    sq.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return png, ico


def arch():
    m = platform.machine().lower()
    return {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(m, m)


def run_pyinstaller(png, ico):
    import PyInstaller.__main__
    sep = os.pathsep
    args = [
        os.path.join(ROOT, "sonic_app.py"),
        "--name", NAME, "--noconfirm", "--clean", "--windowed", "--onedir",
        "--distpath", DIST, "--workpath", os.path.join(BUILD, "pyi"), "--specpath", BUILD,
        "--add-data", f"{os.path.join(ASSETS, 'sonic')}{sep}assets/sonic",
        "--add-data", f"{os.path.join(ASSETS, 'models')}{sep}assets/models",
        "--add-data", f"{png}{sep}assets",
        "--collect-all", "mediapipe",
        "--collect-all", "pyvirtualcam",
        "--icon", ico if IS_WIN else png,
    ]
    if IS_WIN:
        args += ["--hidden-import", "pystray._win32",
                 "--collect-submodules", "pygrabber", "--collect-submodules", "comtypes"]
    if IS_MAC:
        args += ["--osx-bundle-identifier", "com.soniccam.app"]
    PyInstaller.__main__.run(args)


def patch_macos_bundle():
    app = os.path.join(DIST, f"{NAME}.app")
    plist = os.path.join(app, "Contents", "Info.plist")
    with open(plist, "rb") as fp:
        info = plistlib.load(fp)
    info["NSCameraUsageDescription"] = "Sonic Cam utilise votre webcam pour y incruster Sonic."
    info["CFBundleDisplayName"] = "Sonic Cam"
    info["NSHighResolutionCapable"] = True
    with open(plist, "wb") as fp:
        plistlib.dump(info, fp)
    # la modif du plist invalide la signature ad hoc : on re-signe (sans certificat Apple)
    subprocess.check_call(["codesign", "--force", "--deep", "--sign", "-", app])


def package():
    if IS_WIN:
        out = os.path.join(DIST, f"{NAME}-windows-{arch()}")
        shutil.make_archive(out, "zip", DIST, NAME)
        return out + ".zip"
    if IS_MAC:
        out = os.path.join(DIST, f"{NAME}-macos-{arch()}.zip")
        subprocess.check_call(["ditto", "-c", "-k", "--keepParent", os.path.join(DIST, f"{NAME}.app"), out])
        return out
    out = os.path.join(DIST, f"{NAME}-linux-{arch()}.tar.gz")
    with tarfile.open(out, "w:gz") as tar:              # tar.gz garde les droits d'exécution
        tar.add(os.path.join(DIST, NAME), arcname=NAME)
    return out


def main():
    ensure_assets()
    png, ico = make_icon()
    for p in (os.path.join(DIST, NAME), os.path.join(DIST, f"{NAME}.app")):
        shutil.rmtree(p, ignore_errors=True)
    run_pyinstaller(png, ico)
    if IS_MAC:
        patch_macos_bundle()
    print("\nArchive prête :", package())


if __name__ == "__main__":
    main()
