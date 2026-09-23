"""NOIRLab Astro Data Archive downloader.

Ported from the standalone `noirlab_downloader.py` script into the
pipeline's conventions: driven by `AcquisitionConfig` (see `config.py`),
logs through the standard `logging` module instead of `print`, and
raises `AcquisitionError` instead of ad hoc `RuntimeError`s so callers
can catch it alongside `ReductionError` / `AstrometryError` / etc.

This module runs *before* there's a FrameCollection to build -- it's
not a PipelineStage. The usual flow is:

    cfg = PipelineConfig.from_yaml(config_path)
    if cfg.acquisition:
        download_night(cfg.acquisition, outdir=cfg.raw_dir)
    frames = FrameCollection.from_directory(cfg.raw_dir)
    ...

--------------------------------------------------------------------
CREDENTIALS -- your password is never hardcoded in this file.
--------------------------------------------------------------------
Looked for, in this order:
  1. Environment variables  NOIRLAB_EMAIL  /  NOIRLAB_PASSWORD
  2. A local file ~/.noirlab_credentials.json (created once via
     `save_credentials()` / `newhope-pipeline download --set-credentials`,
     saved with permissions restricted to your user only)
  3. An interactive, hidden prompt (nothing gets written to disk)

--------------------------------------------------------------------
NOTE on the authentication header
--------------------------------------------------------------------
The archive's public advanced-search notebook documents the search API
but not the exact header scheme used to authorize *downloads* of
proprietary files. This tries "Bearer <token>" first and falls back to
"Token <token>" on a 401/403, so it should work either way. If every
download still fails after both attempts, the most common real cause
(per the archive's own FAQ) is that your account isn't registered as
PI/Co-I on that proposal yet -- ask the PI to request access via
astroarchive@noirlab.edu.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import requests

from ..core.exceptions import AcquisitionError

if TYPE_CHECKING:
    from ..config import AcquisitionConfig

logger = logging.getLogger(__name__)

NATROOT = "https://astroarchive.noirlab.edu"
ADSURL = f"{NATROOT}/api/adv_search"

# OBSTYPE values (case-insensitive substring match) treated as calibrations
CALIB_KEYWORDS = ["dark", "dflat"]

# Fields requested from the archive for every matched file. Kept to the
# ones demonstrated in the archive's advanced-search.ipynb, to stay
# within "core" (fast) fields and avoid unknown-field errors.
OUTFIELDS = [
    "md5sum",
    "archive_filename",
    "original_filename",
    "instrument",
    "telescope",
    "proc_type",
    "obs_type",
    "proposal",
    "caldat",
    "release_date",
    "url",
]

CREDENTIALS_FILE = Path.home() / ".noirlab_credentials.json"


@dataclass
class DownloadResult:
    """Summary returned by `download_night` / `NoirlabClient.download_all`."""

    science_found: int = 0
    calibrations_found: int = 0
    downloaded: int = 0
    failed: int = 0
    failed_filenames: list[str] = field(default_factory=list)

    @property
    def total_found(self) -> int:
        return self.science_found + self.calibrations_found


# =====================================================================
# Credentials
# =====================================================================
def save_credentials() -> None:
    """Interactive one-time setup: prompt for email/password and store
    them locally (permissions restricted to your user) so future runs
    don't need to ask again."""
    email = input("NOIRLab Astro Data Archive email: ").strip()
    password = getpass.getpass("Password (hidden): ")
    CREDENTIALS_FILE.write_text(json.dumps({"email": email, "password": password}))
    os.chmod(CREDENTIALS_FILE, 0o600)
    logger.info("Saved credentials to %s (owner-read/write only).", CREDENTIALS_FILE)


def get_credentials() -> tuple[str, str]:
    email = os.environ.get("NOIRLAB_EMAIL")
    password = os.environ.get("NOIRLAB_PASSWORD")
    if email and password:
        return email, password

    if CREDENTIALS_FILE.exists():
        data = json.loads(CREDENTIALS_FILE.read_text())
        return data["email"], data["password"]

    logger.info("No stored NOIRLab credentials found.")
    email = input("NOIRLab Astro Data Archive email: ").strip()
    password = getpass.getpass("Password (hidden, not stored): ")
    return email, password


# =====================================================================
# Client
# =====================================================================
class NoirlabClient:
    """Thin, stateful wrapper around the Advanced Search + Retrieve API.

    Holds the authenticated `requests.Session`; the module-level
    functions above only handle credential resolution/storage.
    """

    def __init__(self) -> None:
        self.session: Optional[requests.Session] = None

    def verify_api(self) -> None:
        resp = requests.get(f"{NATROOT}/api/version", timeout=15)
        resp.raise_for_status()
        logger.info("Connected to NOIRLab Astro Data Archive API v%s", resp.text.strip())

    def login(self, email: str, password: str) -> None:
        resp = requests.post(
            f"{NATROOT}/api/get_token/",
            json={"email": email, "password": password},
            timeout=30,
        )
        if resp.status_code != 200:
            raise AcquisitionError(
                f"NOIRLab login failed (HTTP {resp.status_code}). Check your email/password.\n"
                f"Server said: {resp.text[:300]}"
            )
        token = resp.json()
        if isinstance(token, dict):
            # be defensive in case a future API version wraps the token, e.g. {"token": "..."}
            token = token.get("token") or token.get("access_token") or next(iter(token.values()))
        token = str(token).strip()

        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {token}"})
        self.session = session
        logger.info("Login OK.")

    def _require_session(self) -> requests.Session:
        if self.session is None:
            raise AcquisitionError("NoirlabClient.login() must be called before searching/downloading.")
        return self.session

    # ---------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------
    def _run_search(self, search_clauses: list, limit: int, rectype: str = "file") -> list[dict]:
        session = self._require_session()
        query = {"outfields": OUTFIELDS, "search": search_clauses}
        url = f"{ADSURL}/find/?rectype={rectype}&limit={limit}"
        resp = session.post(url, json=query, timeout=120)
        if resp.status_code != 200:
            try:
                msg = resp.json().get("errorMessage", resp.text)
            except Exception:
                msg = resp.text
            raise AcquisitionError(f"NOIRLab search failed (HTTP {resp.status_code}): {msg}")
        return resp.json()[1:]  # first element of the response is query metadata, not a row

    def search_science(
        self, instrument: str, telescope: str, proposal: str, night: str, proctype: str, limit: int
    ) -> list[dict]:
        clauses = [
            ["instrument", instrument],
            ["telescope", telescope],
            ["proposal", proposal],
            ["caldat", night, night],
        ]
        if proctype:
            clauses.append(["proc_type", proctype])
        return self._run_search(clauses, limit)

    def search_calibrations(
        self, instrument: str, telescope: str, night: str, proctype: str, limit: int
    ) -> list[dict]:
        # Archive-wide for that night -- no proposal filter. Calibration
        # frames aren't tied to a single proposal, per the archive's FAQ.
        clauses = [
            ["instrument", instrument],
            ["telescope", telescope],
            ["caldat", night, night],
        ]
        if proctype:
            clauses.append(["proc_type", proctype])
        rows = self._run_search(clauses, limit)
        return [r for r in rows if any(k in str(r.get("obs_type", "")).lower() for k in CALIB_KEYWORDS)]

    # ---------------------------------------------------------------
    # Download
    # ---------------------------------------------------------------
    @staticmethod
    def _md5(path: Path, chunk_size: int = 1 << 20) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()

    def download_file(self, row: dict, dest_dir: Path, chunk_size: int = 1 << 20) -> bool:
        session = self._require_session()
        dest_dir.mkdir(parents=True, exist_ok=True)
        filename = os.path.basename(
            row.get("archive_filename") or row.get("original_filename") or row["md5sum"]
        )
        dest_path = dest_dir / filename

        if dest_path.exists() and row.get("md5sum"):
            if self._md5(dest_path) == row["md5sum"]:
                logger.info("[skip] %s (already downloaded, checksum OK)", filename)
                return True
            logger.info("[redo] %s (exists but checksum mismatch)", filename)

        url = row.get("url") or f"{NATROOT}/api/retrieve/{row['md5sum']}/"

        for attempt in range(2):
            resp = session.get(url, stream=True, timeout=180)
            if resp.status_code == 200:
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size):
                        f.write(chunk)
                size_mb = dest_path.stat().st_size / 1e6
                logger.info("[ok]   %s  (%.1f MB)", filename, size_mb)
                return True
            if resp.status_code in (401, 403) and attempt == 0:
                # First failure: flip the auth scheme once and retry.
                current = session.headers.get("Authorization", "")
                token = current.split(" ", 1)[-1]
                alt_scheme = "Token" if current.startswith("Bearer") else "Bearer"
                session.headers["Authorization"] = f"{alt_scheme} {token}"
                continue
            logger.error("[FAIL] %s  HTTP %s: %s", filename, resp.status_code, resp.text[:200])
            return False
        return False

    def download_all(
        self, rows: list[tuple[str, dict]], outdir: Path
    ) -> DownloadResult:
        result = DownloadResult()
        for r in rows:
            sub = outdir
            if self.download_file(r, sub):
                result.downloaded += 1
            else:
                result.failed += 1
                result.failed_filenames.append(
                    os.path.basename(r.get("archive_filename") or r.get("original_filename") or r["md5sum"])
                )
        return result


# =====================================================================
# High-level entry point -- what cli.py calls
# =====================================================================
def download_night(cfg: AcquisitionConfig, outdir: Path, dry_run: bool = False) -> DownloadResult:
    """Download one proposal's science images + same-night calibrations.

    `cfg` is `PipelineConfig.acquisition` (see `config.py`). Raises
    `AcquisitionError` on login/search failures; per-file download
    failures are instead tallied in the returned `DownloadResult`.
    """
    if not cfg.proposal or not cfg.night:
        raise AcquisitionError("acquisition.proposal and acquisition.night are required to download.")

    client = NoirlabClient()
    client.verify_api()
    email, password = get_credentials()
    logger.info("Logging in to NOIRLab Astro Data Archive...")
    client.login(email, password)

    logger.info(
        "Searching science images: instrument=%s telescope=%s proposal=%s night=%s proc_type=%s",
        cfg.instrument, cfg.telescope, cfg.proposal, cfg.night, cfg.proctype or "(any)",
    )
    science_rows = client.search_science(
        cfg.instrument, cfg.telescope, cfg.proposal, cfg.night, cfg.proctype, cfg.limit
    )
    logger.info("-> %d science files found.", len(science_rows))

    calib_rows: list[dict] = []
    if not cfg.skip_calibrations:
        logger.info("Searching calibration frames (%s) for the same night...", "/".join(CALIB_KEYWORDS))
        calib_rows = client.search_calibrations(
            cfg.instrument, cfg.telescope, cfg.night, cfg.proctype, cfg.limit
        )
        logger.info("-> %d calibration files found.", len(calib_rows))

    all_rows = [("science", r) for r in science_rows] + [("calib", r) for r in calib_rows]
    result = DownloadResult(science_found=len(science_rows), calibrations_found=len(calib_rows))

    if not all_rows:
        logger.warning("Nothing matched -- double-check instrument/telescope/proposal/night.")
        return result

    if dry_run:
        logger.info("DRY RUN -- files that would be downloaded:")
        for kind, r in all_rows:
            logger.info("  [%-7s] obs_type=%-10s %s", kind, str(r.get("obs_type")), r.get("archive_filename"))
        logger.info("Total: %d files. Re-run without dry_run to download.", len(all_rows))
        return result

    outdir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %d files to %s ...", len(all_rows), outdir.resolve())
    download_result = client.download_all(all_rows, outdir)
    result.downloaded = download_result.downloaded
    result.failed = download_result.failed
    result.failed_filenames = download_result.failed_filenames

    logger.info("Done. %d downloaded/verified, %d failed.", result.downloaded, result.failed)
    return result
