"""Download the PUBLIC games we train and validate on. None of this is competition data.

    python training/fetch.py                  # everything, about 2.8 GB
    python training/fetch.py --only metrica   # 190 MB
    python training/fetch.py --only dfl       # 2.6 GB

Two open datasets, both about OTHER matches (which RULES.md allows):

  Metrica Sports sample data, 3 games.  github.com/metrica-sports/sample-data
      Same vendor and file formats as the competition tracking. Has event data, no frame labels.

  DFL open data, 7 Bundesliga matches.  Bassek et al., Scientific Data 2025,
      doi.org/10.6084/m9.figshare.28196177, licence CC-BY 4.0.
      Official tracking at 25 fps with, on EVERY frame, the team in possession and whether the
      ball is in play -- the same kind of label the competition is scored on.

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


def _download(url, path, size=None, md5=None):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", choices=["metrica", "dfl"], help="fetch just one dataset")
    parser.add_argument("--dir", default=PUBLIC_DIR)
    args = parser.parse_args()
    if args.only != "dfl":
        fetch_metrica(args.dir)
    if args.only != "metrica":
        fetch_dfl(args.dir)
    sys.exit(0)
