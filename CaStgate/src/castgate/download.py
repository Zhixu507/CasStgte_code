"""Public Figshare file verified from the dataset DOI landing page.

No archive extraction or fabricated upstream checksum. A local checksum is recorded.
"""

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://figshare.com/ndownloader/files/45206104"
LANDING = "https://doi.org/10.6084/m9.figshare.25459174"


def download_lunghist(output, dry_run=False):
    info = dict(dataset="LungHist700", doi=LANDING, url=URL, file="LungHist700.rar",
                license="CC BY 4.0", approximate_size="639.3 MB (publisher landing page)")
    if dry_run:
        return info
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "LungHist700.rar"
    part = out / "LungHist700.rar.part"
    if target.exists() or part.exists():
        raise FileExistsError(f"Download or partial file already exists in {out}; inspect it before retrying")
    req = urllib.request.Request(URL, headers={"User-Agent": "castgate-research/0.1"})
    digest, total = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(req, timeout=60) as response, part.open("wb") as f:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                f.write(block)
                digest.update(block)
                total += len(block)
                if total % (32 * 1024 * 1024) == 0:
                    print(f"Downloaded {total // (1024*1024)} MiB", flush=True)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Official download failed: {exc}. Download from {LANDING} in a browser; "
                           "the project does not silently substitute a third-party mirror.") from exc
    with part.open("rb") as f:
        if f.read(6) != b"Rar!\x1a\x07":
            raise ValueError("Response is not a RAR archive; retained .part for inspection")
    part.rename(target)
    info.update(bytes=total, sha256=digest.hexdigest(), upstream_checksum_verified=False)
    (out / "download_provenance.json").write_text(json.dumps(info, indent=2))
    return info
