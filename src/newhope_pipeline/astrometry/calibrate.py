"""Per-exposure, per-detector astrometric calibration with SCAMP."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

from ..config import AstrometryConfig
from ..core.exceptions import AstrometryError, PipelineError
from ..core.frame import Extension, Frame, FrameLike, FrameReference
from ..core.stage import PipelineStage
from ..core.state import PipelineState
from . import scamp_wrapper
from .source_extractor import run_sextractor

logger = logging.getLogger(__name__)
_WCS_KEY = re.compile(
    r"^(WCSAXES|WCSNAME|CTYPE\d+[A-Z]?|CRPIX\d+[A-Z]?|CRVAL\d+[A-Z]?|CUNIT\d+[A-Z]?|"
    r"CDELT\d+[A-Z]?|CROTA\d+[A-Z]?|CD\d+_\d+[A-Z]?|PC\d+_\d+[A-Z]?|"
    r"PV\d+_\d+[A-Z]?|PS\d+_\d+[A-Z]?|RADESYS|RADECSYS|EQUINOX|LONPOLE|LATPOLE|"
    r"A_ORDER|B_ORDER|AP_ORDER|BP_ORDER|A_\d+_\d+|B_\d+_\d+|AP_\d+_\d+|BP_\d+_\d+|"
    r"CPDIS\d+|DP\d+|D2IM\d+.*)$"
)
_STRUCTURAL_KEYS = {
    "SIMPLE",
    "XTENSION",
    "BITPIX",
    "NAXIS",
    "PCOUNT",
    "GCOUNT",
    "EXTEND",
    "END",
    "CHECKSUM",
    "DATASUM",
    "EXTNAME",
    "EXTVER",
}
_STRUCTURAL_KEYS.update(f"NAXIS{axis}" for axis in range(1, 10))


def _scamp_card_error(
    path: Path, line_number: int, card: str | bytes, reason: str
) -> AstrometryError:
    logger.error(
        "Invalid SCAMP header %s at line %d, card %r: %s",
        path,
        line_number,
        card,
        reason,
    )
    return AstrometryError(
        f"Invalid SCAMP header {path} at line {line_number}, card {card!r}: {reason}"
    )


def _split_head_line(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith(("\r", "\n")):
        return line[:-1], line[-1:]
    return line, ""


def _comment_start(card: str) -> int | None:
    """Return the FITS comment slash outside any quoted string value."""
    quoted = False
    index = 8
    while index < len(card):
        character = card[index]
        if character == "'":
            if quoted and index + 1 < len(card) and card[index + 1] == "'":
                index += 2
                continue
            quoted = not quoted
        elif character == "/" and not quoted:
            return index
        index += 1
    return None


def _quoted_value_bounds(card: str) -> tuple[int, int] | None:
    """Return the opening and closing quote indices for a standard FITS string."""
    if len(card) <= 8 or card[8] != "=":
        return None
    index = 9
    while index < len(card) and card[index] == " ":
        index += 1
    if index >= len(card) or card[index] != "'":
        return None
    opening = index
    index += 1
    while index < len(card):
        if card[index] == "'":
            if index + 1 < len(card) and card[index + 1] == "'":
                index += 2
                continue
            return opening, index
        index += 1
    return None


def _normalize_scamp_card(card: str, path: Path, line_number: int) -> str:
    """Replace non-ASCII descriptive text while protecting keywords and WCS values."""
    if not any(ord(character) > 127 for character in card):
        return card

    if any(ord(character) > 127 for character in card[:8]):
        raise _scamp_card_error(
            path, line_number, card, "non-ASCII character in the FITS keyword field"
        )

    keyword = card[:8].strip().upper()
    non_ascii_positions = [
        index for index, character in enumerate(card) if ord(character) > 127
    ]
    if keyword in {"COMMENT", "HISTORY", ""}:
        unsafe_positions: list[int] = []
    else:
        comment_start = _comment_start(card)
        descriptive_value = _quoted_value_bounds(card)
        unsafe_positions = []
        for index in non_ascii_positions:
            in_comment = comment_start is not None and index >= comment_start
            in_descriptive_value = (
                descriptive_value is not None
                and descriptive_value[0] <= index <= descriptive_value[1]
            )
            if not in_comment and not in_descriptive_value:
                unsafe_positions.append(index)

    if _WCS_KEY.match(keyword):
        comment_start = _comment_start(card)
        unsafe_positions = [
            index
            for index in non_ascii_positions
            if comment_start is None or index < comment_start
        ]

    if unsafe_positions:
        raise _scamp_card_error(
            path,
            line_number,
            card,
            "non-ASCII character in a FITS keyword or astrometric value",
        )

    # Replacing each Unicode code point one-for-one keeps card columns and all
    # ASCII values intact. Descriptive COMMENT/HISTORY text and FITS comments
    # are not used for the astrometric solution.
    normalized = "".join(
        character if ord(character) <= 127 else "?" for character in card
    )
    logger.warning(
        "Normalized non-ASCII descriptive text in SCAMP header %s at line %d, card %r",
        path,
        line_number,
        card,
    )
    return normalized


def _first_unparseable_card(cards: list[tuple[int, str]]) -> tuple[int, str] | None:
    """Locate the first card Astropy rejects, for actionable parser diagnostics."""
    parsed_cards: list[str] = []
    for line_number, card in cards:
        parsed_cards.append(card)
        try:
            fits.Header.fromstring("\n".join(parsed_cards), sep="\n")
        except Exception:  # noqa: BLE001 - identify the card Astropy rejected
            return line_number, card
    return cards[-1] if cards else None


def _read_scamp_header(path: Path) -> fits.Header:
    """Read a SCAMP text header, sanitizing descriptive UTF-8 text in memory."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.error("Could not read SCAMP header %s: %s", path, exc)
        raise AstrometryError(f"Could not read SCAMP header {path}: {exc}") from exc

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        line_number = len(re.findall(rb"\r\n|\r|\n", raw[: exc.start])) + 1
        raw_lines = re.split(rb"\r\n|\r|\n", raw)
        raw_line = raw_lines[line_number - 1] if len(raw_lines) >= line_number else raw
        raise _scamp_card_error(
            path,
            line_number,
            raw_line,
            f"header is not valid UTF-8 at byte offset {exc.start}",
        ) from exc

    normalized_lines: list[str] = []
    astropy_cards: list[tuple[int, str]] = []
    end_seen = False
    last_card: tuple[int, str] | None = None
    # Only CR/LF delimit records in the SCAMP text format; str.splitlines()
    # also treats Unicode separators inside descriptive text as line breaks.
    lines = re.split(r"(?<=\n)|(?<=\r)(?!\n)", text)
    if lines and lines[-1] == "":
        lines.pop()
    if not text:
        lines = []
    for line_number, line in enumerate(lines, start=1):
        card, line_ending = _split_head_line(line)
        if end_seen:
            if card.strip():
                raise _scamp_card_error(
                    path,
                    line_number,
                    card,
                    "unexpected content after the required END card",
                )
            normalized_lines.append(line)
            continue

        if any(ord(character) > 127 for character in card[:8]):
            raise _scamp_card_error(
                path,
                line_number,
                card,
                "non-ASCII character in the FITS keyword field",
            )
        keyword = card[:8].strip().upper()
        if keyword == "END":
            if card[:8].ljust(8) != "END     " or card[8:].strip():
                raise _scamp_card_error(
                    path, line_number, card, "malformed END card"
                )
            end_seen = True
            normalized_lines.append(line)
            last_card = (line_number, card)
            continue

        normalized = _normalize_scamp_card(card, path, line_number)
        normalized_lines.append(normalized + line_ending)
        astropy_cards.append((line_number, normalized))
        last_card = (line_number, card)

    if not end_seen:
        line_number, card = last_card or (1, "<empty header>")
        raise _scamp_card_error(
            path, line_number, card, "missing required END card"
        )

    try:
        # SCAMP writes its END card as the short text record END followed by
        # spaces. Validate it above; fromstring parses the newline-delimited
        # cards without requiring Astropy's fixed-width END card image.
        return fits.Header.fromstring("".join(normalized_lines), sep="\n")
    except Exception as exc:
        bad_card = _first_unparseable_card(astropy_cards)
        line_number, card = bad_card or last_card or (1, "<empty header>")
        raise _scamp_card_error(
            path, line_number, card, f"Astropy could not parse card: {exc}"
        ) from exc


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2), encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _file_stamp(path: Path | None) -> dict | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.is_file():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _executable_stamp(executable: str) -> dict:
    path = Path(executable).expanduser()
    if path.is_file():
        return _file_stamp(path) or {"command": executable}
    resolved = shutil.which(executable)
    return (
        _file_stamp(Path(resolved))
        if resolved
        else {"command": executable, "unresolved": True}
    )


def _header_true(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().casefold() in {"true", "t", "yes", "y", "1"}


def _valid_wcs(header: fits.Header, shape: tuple[int, ...], label: str) -> WCS:
    try:
        # Older NEWFIRM headers use RADECSYS=FK5; Astropy warns that RADESYS
        # is the modern spelling, but accepts and correctly interprets it.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=r"RADECSYS=", category=FITSFixedWarning
            )
            wcs = WCS(header, relax=True)
            celestial = wcs.celestial
            if not celestial.has_celestial or celestial.pixel_n_dim != 2:
                raise ValueError("header does not contain a 2-D celestial WCS")
            center = [(size - 1) / 2 for size in reversed(shape)]
            world = np.asarray(celestial.pixel_to_world_values(*center), dtype=float)
            scale = np.asarray(celestial.pixel_scale_matrix, dtype=float)
            if not np.all(np.isfinite(world)) or not np.all(np.isfinite(scale)):
                raise ValueError("WCS contains non-finite coordinates or scales")
            if np.linalg.det(scale) == 0:
                raise ValueError("WCS has a singular pixel scale matrix")
        return celestial
    except Exception as exc:
        raise AstrometryError(f"{label} has no usable celestial WCS: {exc}") from exc


def _merged_wcs(
    original: fits.Header, calibrated: fits.Header, shape: tuple[int, ...], label: str
) -> fits.Header:
    if not any(_WCS_KEY.match(key) for key in calibrated):
        raise AstrometryError(f"SCAMP header for {label} contains no WCS keywords")
    merged = original.copy()
    for key in list(merged.keys()):
        if _WCS_KEY.match(key):
            merged.remove(key, remove_all=True, ignore_missing=True)
    for card in calibrated.cards:
        key = card.keyword
        if key in _STRUCTURAL_KEYS or key in {"COMMENT", "HISTORY", ""}:
            continue
        try:
            merged[key] = (card.value, card.comment)
        except Exception as exc:
            raise AstrometryError(
                f"Invalid SCAMP header card {key!r} for {label}: {exc}"
            ) from exc
    _valid_wcs(merged, shape, f"SCAMP result for {label}")
    return merged


def _detector_header(frame: Frame, extension: Extension) -> fits.Header:
    """Flatten common and detector metadata into an ordinary primary image header."""
    header = frame.header.copy()
    for key in list(header.keys()):
        if _WCS_KEY.match(key):
            header.remove(key, remove_all=True, ignore_missing=True)
    for card in extension.header.cards:
        if card.keyword in _STRUCTURAL_KEYS or card.keyword in {
            "COMMENT",
            "HISTORY",
            "",
        }:
            continue
        header[card.keyword] = (card.value, card.comment)
    for key in ("CHECKSUM", "DATASUM"):
        header.remove(key, ignore_missing=True)
    return header


class AstrometricCalibration(PipelineStage):
    """Detect sources and calibrate each detector extension against SCAMP."""

    name = "astrometric_calibration"

    def __init__(
        self,
        config: AstrometryConfig,
        n_workers: int = 1,
        timeout: float | None = None,
        batch_size: int = 1,
    ) -> None:
        super().__init__(
            config,
            n_workers=n_workers,
            timeout=timeout or config.timeout_seconds,
            batch_size=batch_size,
        )

    def _signature(self, frame: FrameLike) -> str:
        cfg = self.config
        payload = {
            "input": _file_stamp(frame.path),
            "extensions": [
                {
                    "index": ext.index,
                    "shape": ext.shape,
                    "detector": ext.header.get("DETECTOR"),
                }
                for ext in frame.extensions
            ],
            "config": cfg.model_dump(mode="json"),
            "files": [
                _file_stamp(cfg.scamp_config_path),
                _file_stamp(cfg.sextractor_config_path),
                _file_stamp(cfg.sextractor_parameters_path),
                _file_stamp(cfg.sextractor_filter_path),
            ],
            "executables": [
                _executable_stamp(path)
                for path in (cfg.sextractor_executable, cfg.scamp_executable)
            ],
            "reference_catalog_path": _file_stamp(cfg.reference_catalog_path),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()

    def _signature_path(self, output_path: Path) -> Path:
        return output_path.with_suffix(output_path.suffix + ".astrometry.json")

    def _restore_cached(
        self, frame: FrameLike, state: PipelineState
    ) -> FrameLike | None:
        if not state.is_done(frame.frame_id, self.name):
            return None
        output = self._output_path(frame)
        if output is None:
            return None
        try:
            metadata = json.loads(
                self._signature_path(output).read_text(encoding="utf-8")
            )
            if (
                metadata.get("signature") != self._signature(frame)
                or not output.is_file()
                or metadata.get("output") != _file_stamp(output)
            ):
                logger.info(
                    "%s: inputs or astrometry settings changed; recalibrating %s",
                    self.name,
                    frame.path.name,
                )
                return None
            cached = FrameReference.from_fits(output, expected_extensions=frame.n_ext)
            if not _header_true(cached.header.get("ASTROMET", False)):
                raise AstrometryError("cached product is missing ASTROMET=True")
            for ext in cached.extensions:
                _valid_wcs(
                    ext.header, ext.shape, f"Cached WCS in {output.name}[{ext.index}]"
                )
        except (OSError, ValueError, TypeError, AstrometryError, PipelineError):
            logger.info(
                "%s: missing cache signature for %s; recalibrating",
                self.name,
                frame.path.name,
            )
            return None
        return super()._restore_cached(frame, state)

    def _write_signature(self, frame: FrameLike, output_path: Path) -> None:
        _json_write(
            self._signature_path(output_path),
            {"signature": self._signature(frame), "output": _file_stamp(output_path)},
        )

    def _persist(self, frame: Frame) -> str | None:
        output = super()._persist(frame)
        if output:
            self._write_signature(frame, Path(output))
        return output

    def _process_item(self, item: FrameLike) -> Frame | FrameReference:
        result = super()._process_item(item)
        if isinstance(result, FrameReference) and self.output_dir is not None:
            self._write_signature(item, result.path)
        return result

    def _intermediate_dir(self, frame: Frame, extension: Extension) -> Path:
        safe_detector = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(extension.header.get("DETECTOR", f"EXT{extension.index}")),
        )
        if self.output_dir is not None:
            path = (
                Path(self.output_dir)
                / "intermediates"
                / frame.frame_id
                / f"detector_{extension.index}_{safe_detector}"
            )
            path.mkdir(parents=True, exist_ok=True)
            return path
        return Path(
            tempfile.mkdtemp(
                prefix=f"newhope-astrometry-{frame.frame_id}-{extension.index}-"
            )
        )

    def _process_extension(self, frame: Frame, extension: Extension) -> fits.Header:
        cfg: AstrometryConfig = self.config
        label = f"{frame.path.name}[{extension.index}:{extension.header.get('DETECTOR', extension.name)}]"
        _valid_wcs(extension.header, extension.shape, f"Initial WCS for {label}")
        work_dir = self._intermediate_dir(frame, extension)
        catalog_path = work_dir / "sources.cat"
        head_path = work_dir / "sources.head"
        catalog_meta = work_dir / "catalog.json"
        head_meta = work_dir / "astrometry.json"
        input_signature = hashlib.sha256(
            json.dumps(
                {
                    "source": _file_stamp(frame.path),
                    "extension": extension.index,
                    "config": cfg.model_dump(mode="json"),
                    "sex_config": _file_stamp(cfg.sextractor_config_path),
                    "sex_parameters": _file_stamp(cfg.sextractor_parameters_path),
                    "sex_filter": _file_stamp(cfg.sextractor_filter_path),
                    "sextractor_executable": _executable_stamp(
                        cfg.sextractor_executable
                    ),
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        catalog_valid = False
        if (
            cfg.reuse_intermediates
            and catalog_path.is_file()
            and catalog_meta.is_file()
        ):
            try:
                catalog_valid = (
                    json.loads(catalog_meta.read_text()).get("signature")
                    == input_signature
                )
            except (OSError, ValueError):
                pass

        if not catalog_valid:
            catalog_path.unlink(missing_ok=True)
            catalog_meta.unlink(missing_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="sextractor-input-", dir=work_dir
            ) as temp_dir:
                image_path = Path(temp_dir) / "detector.fits"
                hdu = fits.PrimaryHDU(
                    data=extension.data, header=_detector_header(frame, extension)
                )
                hdu.writeto(image_path, overwrite=True)
                run_sextractor(
                    image_path,
                    catalog_path,
                    executable=cfg.sextractor_executable,
                    config_path=cfg.sextractor_config_path,
                    parameters_path=cfg.sextractor_parameters_path,
                    filter_path=cfg.sextractor_filter_path,
                    threshold_sigma=cfg.detection_threshold_sigma,
                    minimum_area=cfg.detection_minimum_area,
                    timeout_seconds=cfg.timeout_seconds,
                )
            _json_write(catalog_meta, {"signature": input_signature})

        scamp_signature = hashlib.sha256(
            json.dumps(
                {
                    "catalog": input_signature,
                    "config": _file_stamp(cfg.scamp_config_path),
                    "executable": _executable_stamp(cfg.scamp_executable),
                    "reference_catalog": cfg.reference_catalog,
                    "reference_catalog_path": _file_stamp(cfg.reference_catalog_path),
                    "parameters": cfg.scamp_parameters,
                },
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        head_valid = False
        if cfg.reuse_intermediates and head_path.is_file() and head_meta.is_file():
            try:
                head_valid = (
                    json.loads(head_meta.read_text()).get("signature")
                    == scamp_signature
                )
            except (OSError, ValueError):
                pass
        if not head_valid:
            head_path.unlink(missing_ok=True)
            head_meta.unlink(missing_ok=True)
            (work_dir / "scamp.xml").unlink(missing_ok=True)
            parameters = dict(cfg.scamp_parameters)
            parameters["ASTREF_CATALOG"] = cfg.reference_catalog
            if cfg.reference_catalog_path is not None:
                parameters["ASTREFCAT_NAME"] = str(
                    cfg.reference_catalog_path.expanduser().resolve()
                )
            produced_head = scamp_wrapper.run_scamp(
                catalog_path,
                cfg.scamp_config_path,
                executable=cfg.scamp_executable,
                timeout_seconds=cfg.timeout_seconds,
                parameters=parameters,
            )
            if produced_head.resolve() != head_path.resolve():
                produced_head.replace(head_path)
            _json_write(head_meta, {"signature": scamp_signature})

        scamp_header = _read_scamp_header(head_path)
        merged = _merged_wcs(extension.header, scamp_header, extension.shape, label)
        if not cfg.keep_intermediates or self.output_dir is None:
            shutil.rmtree(work_dir, ignore_errors=True)
        return merged

    def process(self, frame: Frame) -> Frame:
        new_headers = [self._process_extension(frame, ext) for ext in frame.extensions]
        primary = frame.header.copy()
        primary["ASTROMET"] = (True, "Astrometric WCS calibrated with SCAMP")
        primary["ASTRREF"] = (
            self.config.reference_catalog[:68],
            "SCAMP astrometric reference catalog",
        )
        primary["ASTRPIPE"] = ("SCAMP", "Astrometric calibration software")
        return frame.with_extension_headers(new_headers)._replace(
            [
                ext.with_header(header)
                for ext, header in zip(frame.extensions, new_headers)
            ],
            header=primary,
        )


def check_astrometry_tools(config: AstrometryConfig) -> None:
    """Fail before reduction starts if calibration executables/configs are unavailable."""
    for label, executable in (
        ("SExtractor", config.sextractor_executable),
        ("SCAMP", config.scamp_executable),
    ):
        scamp_wrapper.resolve_executable(executable, label)
    for label, path in (
        ("SCAMP config", config.scamp_config_path),
        ("SExtractor config", config.sextractor_config_path),
        ("SExtractor parameter file", config.sextractor_parameters_path),
        ("SExtractor filter file", config.sextractor_filter_path),
    ):
        if not Path(path).expanduser().is_file():
            raise AstrometryError(f"{label} does not exist: {path}")
    scamp_wrapper.validate_reference_catalog(
        config.scamp_executable, config.reference_catalog
    )
    if config.reference_catalog.strip().upper() == "FILE" and (
        config.reference_catalog_path is None
        or not config.reference_catalog_path.expanduser().is_file()
    ):
        raise AstrometryError(
            "astrometry.reference_catalog is FILE, but astrometry.reference_catalog_path "
            "does not point to an existing SCAMP reference catalog"
        )
