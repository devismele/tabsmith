"""Unattended, resumable, verified Slakh2100 archive downloader.

Wraps the verified-resume primitives in :mod:`ml.full_band.slakh` with the
behaviour a multi-hour unattended download needs:

* **Verified resume.** Resumes a ``.download`` temp from its exact current size
  using an HTTP Range request. The server's ``206`` + ``Content-Range`` start
  offset is checked against what we asked for before a single byte is appended;
  anything else aborts rather than corrupting the file by blind append.
* **Retry with backoff.** Transient network failures (timeouts, resets, 5xx,
  429) are retried with exponential backoff. Hard HTTP errors (4xx other than
  408/429) abort.
* **Machine-readable progress.** A JSON file is rewritten atomically every few
  seconds with bytes, percentage, instantaneous/average speed and ETA so the
  run can be observed after terminal output is lost.
* **Integrity gate.** Extraction is never attempted here. The archive is only
  renamed to its final name after the exact byte size *and* the full official
  MD5 both pass. A mismatch quarantines the file outside the repository.

Empirical note: parallel byte-range streams were measured against this record
and gave no throughput benefit (the bottleneck is the local link, not a
per-connection cap), so a single stream is used deliberately.

The dataset root is operator-provided and always outside the repository; no
absolute local path is ever committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from .slakh import SLAKH_ZENODO, SlakhIntegrityError, verify_archive

USER_AGENT = "tabsmith-research/1.0 (dataset acquisition; Slakh2100 CC-BY-4.0)"

# Zenodo's API content endpoint is the identity the record itself advertises.
API_CONTENT_URL = (
    "https://zenodo.org/api/records/{recid}/files/{name}/content".format(
        recid=SLAKH_ZENODO["recid"], name=SLAKH_ZENODO["archiveName"]
    )
)

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass
class Progress:
    state: str
    archive: str
    bytes_done: int
    bytes_total: int
    percent: float
    speed_mb_s: float
    average_mb_s: float
    eta_seconds: float | None
    attempt: int
    updated_at: str
    message: str = ""


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _publish(progress_path: Path, prog: Progress) -> None:
    _write_json_atomic(progress_path, asdict(prog))


def _open_range(session, url: str, offset: int, timeout: int):
    """GET from ``offset``; verify the server actually honoured the range."""
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    resp = session.get(url, headers=headers, stream=True, timeout=timeout)
    if offset:
        if resp.status_code != 206:
            resp.close()
            raise SlakhIntegrityError(
                f"server ignored Range (status {resp.status_code}); refusing append-on-resume"
            )
        content_range = resp.headers.get("Content-Range", "")
        # Expected form: "bytes <start>-<end>/<total>"
        try:
            span = content_range.split()[1].split("/")[0]
            start = int(span.split("-")[0])
        except (IndexError, ValueError):
            resp.close()
            raise SlakhIntegrityError(f"unparsable Content-Range: {content_range!r}")
        if start != offset:
            resp.close()
            raise SlakhIntegrityError(
                f"server resumed at {start}, expected {offset}; refusing append"
            )
    elif resp.status_code != 200:
        resp.close()
        resp.raise_for_status()
    return resp


def download(
    dest_dir: Path,
    *,
    url: str = API_CONTENT_URL,
    expected_size: int = SLAKH_ZENODO["archiveSizeBytes"],
    expected_md5: str = SLAKH_ZENODO["archiveMd5"],
    quarantine_dir: Path | None = None,
    max_attempts: int = 500,
    timeout: int = 120,
    publish_every: float = 5.0,
    session=None,
) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    final = dest_dir / SLAKH_ZENODO["archiveName"]
    partial = final.with_suffix(final.suffix + ".download")
    progress_path = dest_dir / (SLAKH_ZENODO["archiveName"] + ".progress.json")

    if final.exists():
        verify_archive(final, expected_size=expected_size, expected_md5=expected_md5)
        _publish(progress_path, Progress(
            "verified", final.name, expected_size, expected_size, 100.0, 0.0, 0.0, 0,
            0, _now(), "already present and verified",
        ))
        return final

    if session is None:
        import requests  # deferred; only needed for a real download

        session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    run_start = time.time()
    started_bytes = partial.stat().st_size if partial.exists() else 0

    for attempt in range(1, max_attempts + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if have > expected_size:
            raise SlakhIntegrityError(
                f"partial ({have}) larger than official size ({expected_size}); refusing to guess"
            )
        if have == expected_size:
            break

        try:
            resp = _open_range(session, url, have, timeout)
        except SlakhIntegrityError:
            raise
        except Exception as exc:  # transient connection/DNS/timeout
            _publish(progress_path, Progress(
                "retrying", partial.name, have, expected_size,
                100.0 * have / expected_size, 0.0, 0.0, None, attempt, _now(),
                f"connect failed: {type(exc).__name__}: {exc}",
            ))
            time.sleep(min(60.0, 2.0 ** min(attempt, 6)))
            continue

        with resp:
            if resp.status_code in RETRYABLE_STATUS:
                _publish(progress_path, Progress(
                    "retrying", partial.name, have, expected_size,
                    100.0 * have / expected_size, 0.0, 0.0, None, attempt, _now(),
                    f"retryable HTTP {resp.status_code}",
                ))
                time.sleep(min(60.0, 2.0 ** min(attempt, 6)))
                continue
            resp.raise_for_status()

            done = have
            last_publish = 0.0
            window_bytes = 0
            window_start = time.time()
            try:
                with partial.open("ab" if have else "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        window_bytes += len(chunk)
                        now = time.time()
                        if now - last_publish >= publish_every:
                            inst = window_bytes / max(now - window_start, 1e-6) / 1e6
                            avg = (done - started_bytes) / max(now - run_start, 1e-6) / 1e6
                            remaining = expected_size - done
                            eta = remaining / (avg * 1e6) if avg > 0 else None
                            _publish(progress_path, Progress(
                                "downloading", partial.name, done, expected_size,
                                100.0 * done / expected_size, round(inst, 3),
                                round(avg, 3), round(eta, 1) if eta else None,
                                attempt, _now(),
                            ))
                            last_publish = now
                            window_bytes = 0
                            window_start = now
            except Exception as exc:  # mid-stream drop: bytes on disk stay valid
                _publish(progress_path, Progress(
                    "retrying", partial.name, done, expected_size,
                    100.0 * done / expected_size, 0.0, 0.0, None, attempt, _now(),
                    f"stream interrupted: {type(exc).__name__}: {exc}",
                ))
                time.sleep(min(60.0, 2.0 ** min(attempt, 6)))
                continue

        if partial.stat().st_size >= expected_size:
            break
    else:
        raise SlakhIntegrityError(f"exhausted {max_attempts} attempts before completion")

    size = partial.stat().st_size
    if size != expected_size:
        raise SlakhIntegrityError(f"size mismatch after download: {size} != {expected_size}")

    _publish(progress_path, Progress(
        "verifying", partial.name, size, expected_size, 100.0, 0.0, 0.0, None, 0,
        _now(), "computing MD5 over the full archive",
    ))
    digest = hashlib.md5()
    with partial.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            digest.update(block)
    actual = digest.hexdigest()

    if actual != expected_md5:
        quarantine_dir = Path(quarantine_dir or dest_dir.parent / "quarantine")
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = quarantine_dir / f"{partial.name}.{int(time.time())}.badmd5"
        partial.replace(target)
        _publish(progress_path, Progress(
            "failed", target.name, size, expected_size, 100.0, 0.0, 0.0, None, 0,
            _now(), f"MD5 mismatch: {actual} != {expected_md5}; quarantined",
        ))
        raise SlakhIntegrityError(
            f"md5 mismatch: {actual} != {expected_md5} (quarantined to {target.name})"
        )

    partial.replace(final)
    elapsed = time.time() - run_start
    _publish(progress_path, Progress(
        "verified", final.name, size, expected_size, 100.0, 0.0,
        round((size - started_bytes) / max(elapsed, 1e-6) / 1e6, 3), 0.0, 0,
        _now(), f"size and MD5 verified in {elapsed / 3600:.2f} h of this run",
    ))
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.environ.get("SLAKH_ROOT", ""),
        help="Dataset root outside the repository (default: $SLAKH_ROOT).",
    )
    parser.add_argument("--url", default=API_CONTENT_URL)
    parser.add_argument("--max-attempts", type=int, default=500)
    args = parser.parse_args(argv)

    if not args.root:
        parser.error("--root or SLAKH_ROOT is required (must be outside the repository)")

    root = Path(args.root)
    path = download(
        root / "downloads",
        url=args.url,
        quarantine_dir=root / "quarantine",
        max_attempts=args.max_attempts,
    )
    print(f"verified archive: {path.name} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
