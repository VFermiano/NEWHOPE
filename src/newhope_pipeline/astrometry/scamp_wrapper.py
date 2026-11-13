"""Safe subprocess wrapper for SCAMP."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..core.exceptions import AstrometryError, ScampTimeoutError

logger = logging.getLogger(__name__)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ASTREF_LINE = re.compile(r"^\s*ASTREF_CATALOG\s+(\S+)\s+#\s*(.*)$", re.IGNORECASE)
_COMMENT_LINE = re.compile(r"^\s*#\s*(.*)$")
_CATALOG_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*")
_SCAMP_PARAMETER = re.compile(r"^\s*([A-Za-z0-9_]+)\s+(.+?)\s*$")
_SERVER_HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.:-]*$")
_VIZIER_USER_AGENT = "NEWHOPE-pipeline/1.0 (SCAMP reference-catalog relay)"
_VIZIER_FALLBACK_SERVERS = ("vizier.cds.unistra.fr", "vizier.cfa.harvard.edu")


def output_tail(
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    limit: int = 12,
) -> str:
    """Return a compact, ANSI-free subprocess summary for logs and errors."""

    def decode(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    text = _ANSI_ESCAPE.sub("", f"{decode(stdout)}\n{decode(stderr)}")
    lines = [
        line.strip() for line in text.replace("\r", "\n").splitlines() if line.strip()
    ]
    return "\n".join(lines[-limit:])


def resolve_executable(executable: str, tool_name: str) -> str:
    """Resolve a configured executable or raise an actionable pipeline error."""
    candidate = Path(executable).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return str(candidate.resolve())
    else:
        found = shutil.which(executable)
        if found:
            return found
    raise AstrometryError(
        f"Required executable {tool_name!r} was not found ({executable!r}). "
        f"Install {tool_name} or set its executable path in the pipeline configuration."
    )


def parse_supported_reference_catalogs(configuration: str) -> frozenset[str]:
    """Read ASTREF_CATALOG choices from SCAMP's ``-d`` configuration output."""
    lines = configuration.splitlines()
    for index, line in enumerate(lines):
        match = _ASTREF_LINE.match(line)
        if not match:
            continue
        choices = [match.group(1), match.group(2)]
        for continuation in lines[index + 1 :]:
            comment = _COMMENT_LINE.match(continuation)
            if not comment:
                break
            choices.append(comment.group(1))
        return frozenset(
            token.upper()
            for token in _CATALOG_NAME.findall(" ".join(choices))
            if token.casefold() != "or"
        )
    return frozenset()


@lru_cache(maxsize=8)
def supported_reference_catalogs(executable_path: str) -> frozenset[str]:
    """Query one installed SCAMP executable for its supported reference catalogs."""
    try:
        result = subprocess.run(
            [executable_path, "-d"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise AstrometryError(
            f"Could not inspect SCAMP configuration with {executable_path} -d: timed out"
        ) from exc
    except OSError as exc:
        raise AstrometryError(
            f"Could not inspect SCAMP configuration with {executable_path} -d: {exc}"
        ) from exc

    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        details = output_tail(result.stdout, result.stderr, limit=10)
        raise AstrometryError(
            f"Could not inspect supported SCAMP reference catalogs from "
            f"{executable_path} -d (exit {result.returncode})"
            + (f": {details[-1200:]}" if details else "")
        )
    catalogs = parse_supported_reference_catalogs(output)
    if not catalogs:
        raise AstrometryError(
            f"Could not read ASTREF_CATALOG choices from {executable_path} -d output"
        )
    return catalogs


def validate_reference_catalog(executable: str, catalog: str) -> str:
    """Check a configured reference catalog against the selected SCAMP binary."""
    executable_path = resolve_executable(executable, "SCAMP")
    configured = catalog.strip().upper()
    supported = supported_reference_catalogs(executable_path)
    if configured not in supported:
        values = ", ".join(sorted(supported))
        raise AstrometryError(
            f"Unsupported astrometry.reference_catalog {catalog!r} for SCAMP "
            f"({executable_path}). This executable supports: {values}. Set "
            "astrometry.reference_catalog to one of those values."
        )
    return configured


def _configured_parameter(
    config_path: Path,
    parameters: dict[str, str | int | float],
    name: str,
) -> str | None:
    for key, value in parameters.items():
        if key.upper() == name:
            return str(value).strip()
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        uncommented = line.split("#", 1)[0].strip()
        match = _SCAMP_PARAMETER.match(uncommented)
        if match and match.group(1).upper() == name:
            return match.group(2).strip().strip("\"'")
    return None


def _reference_servers(
    config_path: Path,
    parameters: dict[str, str | int | float],
) -> tuple[str, ...]:
    catalog = (
        _configured_parameter(config_path, parameters, "ASTREF_CATALOG") or ""
    ).upper()
    if catalog in {"FILE", "NONE"}:
        return ()
    configured = _configured_parameter(config_path, parameters, "REF_SERVER")
    if not configured:
        return ()
    servers = tuple(
        token
        for token in re.split(r"[\s,]+", configured)
        if token and _SERVER_HOST.fullmatch(token)
    )
    if any(
        server.split(":", 1)[0].casefold() == "localhost"
        or server.startswith("127.")
        for server in servers
    ):
        return ()
    return tuple(dict.fromkeys((*servers, *_VIZIER_FALLBACK_SERVERS)))


def _reference_timeout(
    config_path: Path,
    parameters: dict[str, str | int | float],
    pipeline_timeout: float,
) -> float:
    configured = _configured_parameter(config_path, parameters, "REF_TIMEOUT")
    try:
        seconds = float(configured) if configured is not None else 0
    except ValueError:
        seconds = 0
    return min(seconds, pipeline_timeout) if seconds > 0 else pipeline_timeout


@contextmanager
def _vizier_user_agent_relay(
    servers: tuple[str, ...], timeout_seconds: float
) -> Iterator[str]:
    """Relay SCAMP's loopback HTTP queries to VizieR with an identifying UA.

    SCAMP 2.10 sends VizieR requests without a User-Agent. Some current mirrors
    answer those requests with HTTP 500, while the same query succeeds with a
    normal client header. This short-lived loopback server adds that header and
    leaves the SCAMP catalog query unchanged.
    """

    class RelayHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            failures: list[str] = []
            for host in servers:
                response = None
                for scheme in ("https", "http"):
                    request = Request(
                        f"{scheme}://{host}{self.path}",
                        headers={"User-Agent": _VIZIER_USER_AGENT},
                    )
                    try:
                        response = urlopen(request, timeout=timeout_seconds)
                        break
                    except HTTPError as exc:
                        failures.append(f"{host}: HTTP {exc.code}")
                        exc.close()
                    except (URLError, TimeoutError, OSError) as exc:
                        failures.append(f"{host}: {exc}")
                if response is None:
                    continue
                with response:
                    status = getattr(response, "status", 200)
                    prefix = response.read(4096)
                    if status != 200 or not prefix.startswith(b"#"):
                        failures.append(
                            f"{host}: HTTP {status}, invalid VizieR response "
                            f"{prefix[:120]!r}"
                        )
                        continue
                    response_text = prefix.decode("utf-8", errors="replace")
                    response_lower = response_text.casefold()
                    error_markers = (
                        "#info\terror=",
                        "query_status=error",
                        "no server is available",
                        "couldn't connect with server",
                        "no connection",
                        "error occured",
                        "error occurred",
                    )
                    if any(marker in response_lower for marker in error_markers):
                        error_lines = [
                            line.strip("# \t")
                            for line in response_text.splitlines()
                            if any(marker in line.casefold() for marker in error_markers)
                        ]
                        failures.append(
                            f"{host}: VizieR query error: "
                            f"{'; '.join(error_lines)[:240]}"
                        )
                        continue
                    if not any(
                        line.strip() and not line.startswith("#")
                        for line in response_text.splitlines()
                    ):
                        failures.append(f"{host}: VizieR returned no catalog rows")
                        continue
                    self.send_response(status)
                    for header in ("Content-Type", "Content-Length"):
                        value = response.headers.get(header)
                        if value:
                            self.send_header(header, value)
                    self.end_headers()
                    try:
                        self.wfile.write(prefix)
                        while chunk := response.read(65536):
                            self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        logger.debug("SCAMP closed a VizieR relay response early")
                    return
            logger.error(
                "VizieR relay could not serve the request: %s", "; ".join(failures)
            )
            self.send_error(502, "No configured VizieR mirror returned a catalog response")

        def log_message(self, format: str, *args: object) -> None:
            logger.debug("VizieR relay: " + format, *args)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), RelayHandler)
    except OSError as exc:
        raise AstrometryError(
            f"Could not start the temporary VizieR relay: {exc}"
        ) from exc
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    try:
        thread.start()
        yield f"127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_scamp(
    catalog_path: Path,
    scamp_config_path: Path,
    executable: str = "scamp",
    timeout_seconds: float = 120.0,
    parameters: dict[str, str | int | float] | None = None,
) -> Path:
    """Run SCAMP on one FITS_LDAC catalog and return its calibrated `.head` file.

    SCAMP writes a header sidecar with the same basename as its input catalog.
    The calibrated WCS is merged into the MEF extension by the caller; the
    original sky-subtracted image is never edited.
    """
    catalog_path = Path(catalog_path).resolve()
    config_path = Path(scamp_config_path).expanduser().resolve()
    if not catalog_path.is_file():
        raise AstrometryError(f"SCAMP input catalog does not exist: {catalog_path}")
    if not config_path.is_file():
        raise AstrometryError(
            f"SCAMP configuration file does not exist: {config_path}. "
            "Set astrometry.scamp_config_path to an installed SCAMP configuration."
        )

    executable_path = resolve_executable(executable, "SCAMP")
    command_parameters = dict(parameters or {})
    for key, value in tuple(command_parameters.items()):
        if key.upper() == "ASTREF_CATALOG":
            command_parameters[key] = validate_reference_catalog(
                executable_path, str(value)
            )
    reference_servers = _reference_servers(config_path, command_parameters)
    relay = (
        _vizier_user_agent_relay(
            reference_servers,
            _reference_timeout(config_path, command_parameters, timeout_seconds),
        )
        if reference_servers
        else nullcontext(None)
    )
    try:
        with relay as local_server:
            if local_server:
                for key in tuple(command_parameters):
                    if key.upper() == "REF_SERVER":
                        del command_parameters[key]
                command_parameters["REF_SERVER"] = local_server
                logger.debug(
                    "Relaying SCAMP VizieR requests to %s with a User-Agent",
                    ", ".join(reference_servers),
                )
            command = [executable_path, str(catalog_path), "-c", str(config_path)]
            for key, value in command_parameters.items():
                if not key.replace("_", "").isalnum():
                    raise AstrometryError(f"Invalid SCAMP parameter name: {key!r}")
                command.extend((f"-{key.upper()}", str(value)))

            logger.info("Running SCAMP for %s", catalog_path.name)
            logger.debug("SCAMP command: %s", command)
            result = subprocess.run(
                command,
                cwd=catalog_path.parent,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired as exc:
        details = output_tail(exc.stdout, exc.stderr)
        raise ScampTimeoutError(
            f"SCAMP exceeded {timeout_seconds:g}s for {catalog_path.name}"
            + (
                f". Check the VizieR connection (REF_SERVER) or use a local "
                f"reference catalog (ASTREF_CATALOG=FILE). Last SCAMP output:\n{details}"
                if details
                else ". Check the VizieR connection (REF_SERVER) or use a local "
                "reference catalog (ASTREF_CATALOG=FILE)."
            )
        ) from exc
    except OSError as exc:
        raise AstrometryError(
            f"Could not start SCAMP ({executable_path}): {exc}"
        ) from exc

    summary = output_tail(result.stdout, result.stderr)
    if summary:
        logger.info("SCAMP output for %s:\n%s", catalog_path.name, summary)
    if result.returncode != 0:
        details = output_tail(result.stderr, result.stdout, limit=20)
        raise AstrometryError(
            f"SCAMP failed for {catalog_path.name} (exit {result.returncode})"
            + (f": {details[-2000:]}" if details else "")
        )

    header_path = catalog_path.with_suffix(".head")
    if not header_path.is_file() or header_path.stat().st_size == 0:
        raise AstrometryError(
            f"SCAMP completed for {catalog_path.name} but did not produce a non-empty "
            f"astrometric header at {header_path}"
        )
    return header_path
