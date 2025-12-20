"""
Prepare a single artist library:
- Normalize filenames to: "{track:02d} - {title}.mp3" (using ffprobe ID3 tags when available)
- Validate each mp3 is decodable (ffmpeg full decode)
- Run MuQ-MuLan embedding extraction and cache to data/cache/{artist}.pkl
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import subprocess
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Path setup (align with scripts/preprocess.py)
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../scripts
subproject_root = os.path.abspath(os.path.join(current_dir, ".."))  # .../genres_in_genres
sys.path.insert(0, subproject_root)  # For src.xxx


_ILLEGAL_CHARS = r'\/:*?"<>|'


def _sanitize_component(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    # Replace illegal filename chars (Windows-safe; also avoids path separators)
    trans = {ord(c): " " for c in _ILLEGAL_CHARS}
    s = s.translate(trans)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_track_number(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    raw = raw.strip()
    # common forms: "08", "8", "08/10", "8/10"
    raw = raw.split("/")[0].strip()
    m = re.match(r"^\d+$", raw)
    if not m:
        return None
    n = int(raw)
    return n if n > 0 else None


def _ffprobe_tags(path: str) -> Dict[str, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format_tags=artist,album,title,track,TRACKNUMBER,date",
        "-of",
        "default=nw=1",
        path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        return {}
    tags: Dict[str, str] = {}
    for line in p.stdout.splitlines():
        # TAG:key=value
        if not line.startswith("TAG:"):
            continue
        kv = line[4:]
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        tags[k.strip().lower()] = v.strip()
    return tags


def _desired_filename_from_tags(fpath: str) -> Optional[str]:
    tags = _ffprobe_tags(fpath)
    title = _sanitize_component(tags.get("title") or "")
    track = _parse_track_number(tags.get("track") or tags.get("tracknumber"))

    if not title:
        return None

    if track is not None:
        return f"{track:02d} - {title}.mp3"
    return f"{title}.mp3"


def _iter_mp3_files(artist_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(artist_dir):
        for fn in files:
            if fn.lower().endswith(".mp3"):
                yield os.path.join(root, fn)


@dataclass(frozen=True)
class RenameOp:
    src: str
    dst: str


def plan_renames(artist_dir: str) -> List[RenameOp]:
    ops: List[RenameOp] = []
    for fpath in sorted(_iter_mp3_files(artist_dir)):
        desired = _desired_filename_from_tags(fpath)
        if not desired:
            continue
        dst = os.path.join(os.path.dirname(fpath), desired)
        if os.path.abspath(dst) == os.path.abspath(fpath):
            continue
        ops.append(RenameOp(src=fpath, dst=dst))
    return ops


def apply_renames(ops: List[RenameOp], dry_run: bool) -> Tuple[int, List[str]]:
    """
    Returns (num_renamed, conflicts)
    Conflicts include destinations that already exist (and are not the same file).
    """
    conflicts: List[str] = []

    # Detect dst collisions
    dsts = [os.path.abspath(op.dst) for op in ops]
    if len(dsts) != len(set(dsts)):
        # multiple sources want same dst
        conflicts.append("Multiple files map to the same destination filename. Aborting.")
        return 0, conflicts

    # Detect existing dst
    for op in ops:
        if os.path.exists(op.dst):
            conflicts.append(f"Exists: {op.dst}")
    if conflicts:
        return 0, conflicts

    if dry_run:
        return 0, []

    renamed = 0
    for op in ops:
        os.rename(op.src, op.dst)
        renamed += 1
    return renamed, []


def _ffmpeg_decode_ok(path: str) -> Tuple[str, bool, str]:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ok = p.returncode == 0
    return path, ok, (p.stderr.strip() if not ok else "")


def validate_mp3s(artist_dir: str, workers: int = 4) -> List[Tuple[str, str]]:
    """
    Returns list of (path, error) for failures.
    """
    files = list(sorted(_iter_mp3_files(artist_dir)))
    failures: List[Tuple[str, str]] = []
    if not files:
        return failures

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(_ffmpeg_decode_ok, p): p for p in files}
        for fut in as_completed(futs):
            path, ok, err = fut.result()
            if not ok:
                failures.append((path, err))
    failures.sort(key=lambda x: x[0])
    return failures


def encode_to_cache(
    artist: str,
    data_dir: str,
    cache_dir: str,
    device: str,
) -> None:
    raise RuntimeError(
        "MuQ-MuLan embedding extraction has been removed from this project. "
        "Use precomputed caches under data/cache/music or plug in your own extractor."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize filenames, validate mp3s, and preprocess a single artist to cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artist", default="周杰伦", help="Artist directory name under data/music/")
    parser.add_argument("--data_dir", default=os.path.join(subproject_root, "data", "music"))
    parser.add_argument("--cache_dir", default=os.path.join(subproject_root, "data", "cache"))
    parser.add_argument("--device", default="cuda", help="Device for MuQ-MuLan encoder (cuda/cpu)")

    parser.add_argument("--dry_run", action="store_true", help="Only print planned renames")
    parser.add_argument("--no_rename", action="store_true", help="Skip renaming step")
    parser.add_argument("--verify", action="store_true", help="Validate mp3 decodability via ffmpeg")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for validation")
    parser.add_argument(
        "--encode",
        action="store_true",
        help="(Disabled) MuQ-MuLan extraction was removed; caches in data/cache/music are precomputed.",
    )
    args = parser.parse_args()

    artist_dir = os.path.join(args.data_dir, args.artist)
    if not os.path.isdir(artist_dir):
        raise FileNotFoundError(f"Artist directory not found: {artist_dir}")

    if not args.no_rename:
        ops = plan_renames(artist_dir)
        if ops:
            print(f"[prepare_artist] Planned renames: {len(ops)}")
            for op in ops:
                print(f"  - {os.path.relpath(op.src, args.data_dir)} -> {os.path.relpath(op.dst, args.data_dir)}")
            renamed, conflicts = apply_renames(ops, dry_run=args.dry_run)
            if conflicts:
                print("[prepare_artist] Conflicts detected; no files renamed:")
                for c in conflicts:
                    print(f"  - {c}")
                return 2
            if args.dry_run:
                print("[prepare_artist] dry_run enabled; no files renamed.")
            else:
                print(f"[prepare_artist] Renamed {renamed} files.")
        else:
            print("[prepare_artist] No renames needed (all filenames already normalized or tags missing).")

    if args.verify:
        print("[prepare_artist] Validating mp3 files (full decode via ffmpeg)...")
        fails = validate_mp3s(artist_dir, workers=args.workers)
        if fails:
            print(f"[prepare_artist] Validation FAILED for {len(fails)} file(s):")
            for p, err in fails:
                rel = os.path.relpath(p, args.data_dir)
                print(f"  - {rel}")
                if err:
                    print(f"    {err.splitlines()[0]}")
            return 3
        print("[prepare_artist] Validation OK (all mp3 decoded successfully).")

    if args.encode:
        print(
            "[prepare_artist] Embedding extraction is disabled (MuQ-MuLan dependency removed).\n"
            "Please use the precomputed caches under data/cache/music or provide a custom extractor."
        )
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
