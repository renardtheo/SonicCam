# Sonic Cam : la caméra qui claque

Votre webcam, avec un Sonic qui débarque au hasard **toutes les 5 à 15 minutes**. Il passe derrière vous ou devant, saute sur votre épaule et vous met une claque : la tête part, l'image tremble, la joue rougit. Ou alors il passe juste discrètement en arrière-plan.

## Pour les utilisateurs : installer l'appli

**Il faut dans tous les cas une caméra virtuelle**, c'est par elle que Teams et Discord reçoivent l'image :

| Système | Caméra virtuelle | Appli Sonic Cam |
|---|---|---|
| **Windows 10/11** | Installer [OBS Studio](https://obsproject.com). Rien d'autre à faire. | Dézipper `SonicCam-windows-x64.zip`, lancer `SonicCam\SonicCam.exe` |
| **macOS** (puce Apple M1 et +) | Installer OBS, cliquer **une fois** sur « Démarrer la caméra virtuelle » (pour installer l'extension), l'arrêter, fermer OBS | Dézipper `SonicCam-macos-arm64.zip`, glisser **SonicCam** dans Applications. 1er lancement : **clic droit > Ouvrir** (appli non signée), puis autoriser la caméra |
| **Linux** (Ubuntu 22.04+) | `sudo apt install v4l2loopback-dkms` puis `sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="Sonic Cam"` (à refaire après chaque redémarrage) | `tar xzf SonicCam-linux-x64.tar.gz` puis `./SonicCam/SonicCam` |

**Ensuite :** dans Teams ou Discord, choisissez la caméra **« OBS Virtual Camera »** (ou « Sonic Cam » sous Linux). Le nom exact est affiché en bas de la fenêtre de l'appli.

### La fenêtre de l'appli
- **Aperçu** de ce qu'envoie la caméra, et le compte à rebours avant le prochain passage.
- **Claque maintenant** / **Passage** pour déclencher Sonic à la main. **Pause** arrête les apparitions, mais la vidéo continue de passer.
- **Webcam** : choix de la vraie caméra. « Automatique » ignore les caméras virtuelles et celles qui renvoient une image noire.
- **Fréquence** : 5 à 15 min par défaut. Il y a aussi un mode « Test : 10 à 30 s ».
- **Effets** : discrets, normaux ou forts. **Image miroir** en option.
- **Windows** : fermer la fenêtre laisse Sonic Cam tourner près de l'horloge. Clic droit sur l'icône pour quitter.

Les réglages et le journal (`sonic_cam.log`, utile en cas de souci) sont enregistrés ici :
- Windows : `%APPDATA%\SonicCam`
- macOS : `~/Library/Application Support/SonicCam`
- Linux : `~/.config/SonicCam`

### Problèmes courants
- **Image noire dans Teams** : l'appli Sonic Cam est-elle lancée ? Si OBS est ouvert avec « Démarrer la caméra virtuelle » actif, arrêtez-la.
- **« Aucune webcam utilisable »** : la webcam est déjà prise par Teams ou Discord. Mettez-y la caméra virtuelle à la place.
- **Windows bloque l'appli** : voir ci-dessous.

### Windows bloque l'exe (SmartScreen / Contrôle intelligent des applications)
L'appli n'est pas signée numériquement, donc Windows se méfie :
- **SmartScreen** (« Windows a protégé votre ordinateur ») : cliquez sur *Informations complémentaires* puis *Exécuter quand même*.
- **Contrôle intelligent des applications** (« Une stratégie de contrôle d'application a bloqué ce fichier ») : il n'y a **pas de contournement** pour une seule appli. Il faut signer l'exe (voir « Signature » plus bas) ou utiliser la version Python (`installer.bat`, puis `lancer_sonic_cam.bat`).

## Pour l'administrateur : fabriquer les versions

PyInstaller ne fait pas de compilation croisée : chaque version se construit **sur son propre système**.

**Option 1, la plus simple : GitHub Actions** (fabrique les 3 versions automatiquement)
1. Créez un dépôt GitHub (privé, c'est possible) et poussez-y ce dossier, **avec le dossier `assets/`** : il contient les sprites et les modèles, donc la vidéo n'est pas nécessaire.
2. Allez dans l'onglet **Actions**, ouvrez **Build Sonic Cam** et cliquez sur **Run workflow**. Au bout d'environ 10 min, les 3 archives sont téléchargeables en bas de la page du run.
3. Pour publier une version, poussez un tag : `git tag v1.0 && git push --tags`. Une *Release* est alors créée avec les 3 archives.

Chaque version est lancée en mode `--selftest` sur la machine de compilation : si elle ne démarre pas, le build échoue.

**Option 2, en local :** `pip install -r requirements.txt -r requirements-build.txt` puis `python build.py` (Python 3.12). L'archive est créée dans `dist/`.

### Signature (recommandée pour une diffusion en entreprise)
- **Windows** : un certificat de signature de code supprime les blocages. La solution la moins chère est *Azure Trusted Signing* (environ 10 $/mois). Sinon, un certificat OV classique.
- **macOS** : un compte Apple Developer (99 $/an) permet de signer et notariser l'appli, ce qui supprime le « clic droit > Ouvrir ».

## Version script (développement)
- `installer.bat` puis `lancer_sonic_cam.bat` (Windows), ou `python sonic_cam.py` partout.
- Options : `--camera`, `--min-gap`, `--max-gap` (en secondes), `--size`, `--impact`, `--mirror`, `--simulate`, `--list-cameras`, `--choose-camera`.
- `python sonic_app.py` lance l'appli avec fenêtre, sans la compiler.

## Fichiers
| Fichier | Rôle |
|---|---|
| `sonic_app.py` | Appli avec fenêtre (point d'entrée des versions compilées) |
| `sonic_cam.py` | Moteur : vision, chorégraphie, effets, caméra virtuelle, plus la ligne de commande |
| `prepare_sonic.py` | Détoure la vidéo et télécharge les modèles (déjà fait : voir `assets/`) |
| `build.py` | Fabrique l'appli pour l'OS courant |
| `.github/workflows/build.yml` | Fabrique les 3 versions sur GitHub |
