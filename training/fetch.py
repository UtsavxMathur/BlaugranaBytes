"""Download the PUBLIC games we train and validate on. None of this is competition data.

    python training/fetch.py                  # everything, about 4.7 GB
    python training/fetch.py --only metrica   # 190 MB
    python training/fetch.py --only dfl       # 2.6 GB
    python training/fetch.py --only skillcorner   # 1.8 GB

Three open datasets, all about OTHER matches (which RULES.md allows):

  Metrica Sports sample data, 3 games.  github.com/metrica-sports/sample-data
      Same vendor and file formats as the competition tracking. Has event data, no frame labels.

  DFL open data, 7 Bundesliga matches.  Bassek et al., Scientific Data 2025,
      doi.org/10.6084/m9.figshare.28196177, licence CC-BY 4.0.
      Official tracking at 25 fps with, on EVERY frame, the team in possession and whether the
      ball is in play -- the same kind of label the competition is scored on.

  SkillCorner open data, 20 A-League matches of 2024/25.  github.com/SkillCorner/opendata,
      MIT licence. Broadcast tracking at 10 fps with the ball (and its height), the player and
      team in possession on every frame, and phases of play that cover exactly the time the
      ball is in play.

Files land in .cache/public/ (gitignored). Set PUBLIC_DATA_DIR to put them elsewhere. A file that
is already complete is not downloaded again.
"""

import argparse
import hashlib
import os
import ssl
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PUBLIC_DIR = os.environ.get("PUBLIC_DATA_DIR") or os.path.join(ROOT, ".cache", "public")

# -- Metrica: name -> size in bytes (from the GitHub tree listing) ---------------------------------
METRICA_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
METRICA = {
    1: {"Sample_Game_1_RawEventsData.csv": 139446,
        "Sample_Game_1_RawTrackingData_Home_Team.csv": 32782961,
        "Sample_Game_1_RawTrackingData_Away_Team.csv": 32782073},
    2: {"Sample_Game_2_RawEventsData.csv": 153897,
        "Sample_Game_2_RawTrackingData_Home_Team.csv": 31890338,
        "Sample_Game_2_RawTrackingData_Away_Team.csv": 29627711},
    3: {"Sample_Game_3_events.json": 1709486,
        "Sample_Game_3_metadata.xml": 81806,
        "Sample_Game_3_tracking.txt": 53443115},
}

# -- DFL: match id -> {kind: (figshare file id, md5)} (from the figshare API for the article) ------
DFL_URL = "https://ndownloader.figshare.com/files/{}"
DFL = {
    "J03WMX": {"info": (51643475, "bc5d5b21dbcf36aa30f4cd0428309f6b"),
               "events": (51643493, "2d88e180a1f63c7e4157feb980fc7255"),
               "positions": (51643514, "e4273a2adf5dbd9d4ec42def2f01905c")},
    "J03WN1": {"info": (51643472, "37b0f85588c0acfb995fe633c7ac2e0d"),
               "events": (51643496, "83e8ed7ea8e8ebbbdc61f61ac3534008"),
               "positions": (51643517, "bacb2f609c1d1b0bfc906f6b9d4109f2")},
    "J03WOH": {"info": (51643478, "b9710781d400dd812f62b0a6e3d7bce8"),
               "events": (51643499, "ab66e51512f39d06f49f27dd3b4c2f40"),
               "positions": (51643520, "3efff25a86a61768e7d253b4bba137a3")},
    "J03WOY": {"info": (51643481, "7b021e577fca917a63111d55cc29ae5b"),
               "events": (51643502, "64b1be01d6ff6848168efebd56f344b9"),
               "positions": (51643523, "4e1063e0a0ae499920cf68ac1aa3ab4e")},
    "J03WPY": {"info": (51643487, "65a4568e4f374af94d3c1015cbc7c4f5"),
               "events": (51643505, "f68482441d38723d69f244f896482803"),
               "positions": (51643526, "e56ff1336a334f8da51110fbe7cab764")},
    "J03WQQ": {"info": (51643484, "5979209af316a914ce57347d0633df6c"),
               "events": (51643508, "b7e9d82af4034539d802319be6c57776"),
               "positions": (51643529, "9105bad987191d3c3c362c1fce9a4fcd")},
    "J03WR9": {"info": (51643490, "5e2d580d3d8ed97e32959de90d04992b"),
               "events": (51643511, "c13bc5efa5eca4a022cc678210b87d5f"),
               "positions": (51643532, "e060bb7d7a5a3bfbd15bbf9bc631024d")},
}


# -- SkillCorner: match id -> (tracking sha256, tracking bytes, match.json bytes, phases bytes),
# pinned to one commit of github.com/SkillCorner/opendata (MIT licence). The tracking files are
# stored with Git LFS, so they come from media.githubusercontent.com; the sha256 is the LFS oid.
SKILLCORNER_COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
SKILLCORNER_RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/data/matches/{}/{}_{}"
SKILLCORNER_LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/data/matches/{}/{}_{}"
SKILLCORNER = {
    1874553: ("ea97f58f8eaad925feaeacc6395ec24860dd80027af8a89450276adebd29d265", 90729279, 33730, 115978),
    1886347: ("3577e2803da95390f8b2f85d47829eb55fbbfee6211c94adc65db0dc78e46b41", 89280839, 27541, 121864),
    1899585: ("90b6289ec2181192d1793ec6240fbbe8225ff0219c7956710c6278c31a661100", 85550410, 27810, 119735),
    1925299: ("48528be40ad82fa3e2b232e1af8d311705f7d6d119925ef199d2b59ea36fd3c9", 97091519, 28396, 131104),
    1927964: ("858c7a756ca3d33017f864775af5999d2c0f364e58ab60ae040c8b9ebab17827", 91049828, 33400, 117263),
    1953632: ("1c2014f97c3eff1ce2475203512c9d126930d7599c036105c9ac442d1de1667e", 90027965, 27767, 114478),
    1959846: ("5ba2d87f6fd1ba3138301b369bd9266be69416a2d799a3fba5b5ec8a52e1284b", 87710167, 32946, 128571),
    1986691: ("630e946950fe459f5e48872b01a36e6dfc319cf9027e7b98156150bb6e2c30f6", 88531597, 33792, 115267),
    1996435: ("f42764d8bde290e8f738fa21141cadbed75408314b0b337bbf1d5a3444540a02", 88634016, 27830, 120335),
    1996436: ("fab9d331e06f3042c83ac17798901b49bb259c99a8f3b319cd0146263cde65e7", 88945845, 32859, 104584),
    2006229: ("3ee193ba01f1fe2cbce087eca8a881c21b8b0ea3485ae6c3d7e32e8b2f51e738", 90078007, 28213, 122340),
    2006363: ("61dfdbd83f8e654127b249b8503f062338c14ec0c9e65404bcf70fb7c1346bd4", 90591887, 33294, 119862),
    2007448: ("12d00d6e4257218812df6e4c4263c49387ba0313a83ff8d826c4416828f9c798", 90025383, 32670, 113871),
    2007721: ("fea913b8d4cbde8d85c6bdf5f1a4b3fd203b78227e976a171c8ab0946705d378", 92872706, 33745, 116997),
    2010085: ("04562d9754b2977c80430535e9b5733426bc4ce910db56fe7a3760876211ac5a", 92516156, 33744, 115871),
    2011166: ("8ec2937050930d87e4f52b2d65515f70ef8edd0ae2814bf40155c4448133390e", 87441570, 28420, 121175),
    2013725: ("13eb83329814ed5f5d73da8f8ab5c5979d6c8a7858e89e28a7fa8072f91fb250", 96059074, 27899, 130821),
    2015213: ("413990d0024d84856ce1f2a16866b8887472c0cea22c61caef0d1f46505a609e", 102030769, 27937, 134493),
    2016236: ("d78555f9dfd7d1ce9de38066d3c98e6d54417c6d45d73b1f1ce7e26f07263d8f", 94737872, 33791, 133630),
    2017461: ("59cfa5588efee041da439f44df5b5b92f2b7305571dc831164e3d05f7415787a", 89543442, 28545, 118313),
}
SKILLCORNER_FILES = ("match.json", "phases_of_play.csv", "tracking_extrapolated.jsonl")


def skillcorner_path(match, name, root=PUBLIC_DIR):
    return os.path.join(root, "skillcorner", str(match), "{}_{}".format(match, name))


def metrica_path(game, name, root=PUBLIC_DIR):
    return os.path.join(root, "metrica", "Sample_Game_{}".format(game), name)


def dfl_path(match, kind, root=PUBLIC_DIR):
    return os.path.join(root, "dfl", "{}_{}.xml".format(match, kind))


def _ssl_context():
    """HTTPS context. python.org's macOS build ships without root certificates; certifi has them."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url, path, size=None, md5=None, sha256=None):
    """Fetch url to path unless a complete copy is already there. Checks size and/or md5."""
    name = os.path.basename(path)
    if os.path.isfile(path) and (size is None or os.path.getsize(path) == size):
        print("  have  {}".format(name))
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    partial = path + ".part"
    request = urllib.request.Request(url, headers={"User-Agent": "public-data-fetch"})
    try:
        with urllib.request.urlopen(request, context=_ssl_context()) as response, \
                open(partial, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            print("  get   {}  ({:.1f} MB)".format(name, total / 1e6), flush=True)
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except urllib.error.URLError as error:
        if "CERTIFICATE_VERIFY_FAILED" in str(error):
            raise SystemExit("SSL certificate check failed. Run `pip install certifi` and retry.")
        raise
    if size is not None and os.path.getsize(partial) != size:
        raise IOError("{}: expected {} bytes, got {}".format(name, size, os.path.getsize(partial)))
    if md5 is not None and _md5(partial) != md5:
        raise IOError("{}: md5 mismatch, the download is corrupt. Delete it and retry.".format(name))
    if sha256 is not None and _sha256(partial) != sha256:
        raise IOError("{}: sha256 mismatch, the download is corrupt. Delete it and retry."
                      .format(name))
    os.replace(partial, path)


def fetch_metrica(root=PUBLIC_DIR, games=None):
    for game in games or sorted(METRICA):
        print("Metrica sample game {}".format(game))
        for name, size in METRICA[game].items():
            url = "{}/Sample_Game_{}/{}".format(METRICA_URL, game, name)
            _download(url, metrica_path(game, name, root), size=size)


def fetch_dfl(root=PUBLIC_DIR, matches=None):
    for match in matches or sorted(DFL):
        print("DFL match {}".format(match))
        for kind, (file_id, md5) in DFL[match].items():
            _download(DFL_URL.format(file_id), dfl_path(match, kind, root), md5=md5)


def fetch_skillcorner(root=PUBLIC_DIR, matches=None):
    for match in matches or sorted(SKILLCORNER):
        print("SkillCorner match {}".format(match))
        sha, size, match_size, phases_size = SKILLCORNER[match]
        for name, file_size in (("match.json", match_size), ("phases_of_play.csv", phases_size)):
            _download(SKILLCORNER_RAW.format(SKILLCORNER_COMMIT, match, match, name),
                      skillcorner_path(match, name, root), size=file_size)
        _download(SKILLCORNER_LFS.format(SKILLCORNER_COMMIT, match, match,
                                         "tracking_extrapolated.jsonl"),
                  skillcorner_path(match, "tracking_extrapolated.jsonl", root), size=size,
                  sha256=sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["metrica", "dfl", "skillcorner"],
                        help="fetch just one dataset")
    parser.add_argument("--dir", default=PUBLIC_DIR)
    args = parser.parse_args()
    if args.only in (None, "metrica"):
        fetch_metrica(args.dir)
    if args.only in (None, "dfl"):
        fetch_dfl(args.dir)
    if args.only in (None, "skillcorner"):
        fetch_skillcorner(args.dir)
    sys.exit(0)
