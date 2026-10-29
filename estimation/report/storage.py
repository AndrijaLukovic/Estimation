"""Lossless, bounded report storage that also works over file:// URLs."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile

MAX_FILE_BYTES = 25_000_000
PART_BYTES = 20_000_000
MAX_REPORT_FILES = 10_000
CALL_PREFIX = "window.ReportData.add("


class DataBundles:
    """Pack serialized JSON into script bundles; large records span bundles."""
    def __init__(self, root, stem="records", part_bytes=PART_BYTES):
        if part_bytes < 512 or part_bytes > MAX_FILE_BYTES:
            raise ValueError("Invalid report bundle size")
        self.root = Path(root)
        self.stem = stem
        self.part_bytes = part_bytes
        self.directory = self.root / "data"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.size = 0
        self.number = 0
        self.keys = set()

    def _next_file(self):
        self.close()
        self.number += 1
        self.relative = f"data/{self.stem}-{self.number:04d}.js"
        self.file = (self.root / self.relative).open("wb")
        self.size = 0

    def add(self, key, source):
        if key in self.keys:
            raise ValueError(f"Duplicate report record key: {key}")
        self.keys.add(key)
        # A quoted character needs at most 6 bytes (including JSON escapes).
        width = max(1, min(256_000, (self.part_bytes - 256) // 12))
        fragments = [source[i:i+width] for i in range(0, len(source), width)] or [""]
        parts = []
        for index, fragment in enumerate(fragments):
            arguments = json.dumps([key, index, fragment], ensure_ascii=False, separators=(",", ":"))
            line = (CALL_PREFIX + arguments[1:-1] + ");\n").encode("utf-8")
            if len(line) > self.part_bytes:
                raise ValueError("Report fragment exceeds its byte budget")
            if self.file is None or self.size + len(line) > self.part_bytes:
                self._next_file()
            self.file.write(line)
            self.size += len(line)
            if not parts or parts[-1] != self.relative:
                parts.append(self.relative)
        encoded = source.encode("utf-8")
        return dict(parts=parts, chunks=len(fragments), bytes=len(encoded),
                    sha256=hashlib.sha256(encoded).hexdigest())

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def iter_bundle_sources(root, descriptors):
    """Read and verify each record without loading the whole report in memory."""
    files = dict.fromkeys(part for item in descriptors.values() for part in item["parts"])
    fragments = {}
    seen = set()
    for relative in files:
        with (Path(root) / relative).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith(CALL_PREFIX) or not line.endswith(");\n"):
                    raise ValueError(f"Invalid data bundle: {relative}")
                key, index, fragment = json.loads("[" + line[len(CALL_PREFIX):-3] + "]")
                if key not in descriptors:
                    continue
                if key in seen:
                    raise ValueError(f"Duplicate report data: {key}")
                pieces = fragments.setdefault(key, [])
                if index != len(pieces):
                    raise ValueError(f"Missing or duplicate fragment: {key}")
                pieces.append(fragment)
                if len(pieces) == descriptors[key]["chunks"]:
                    source = "".join(fragments.pop(key))
                    if hashlib.sha256(source.encode("utf-8")).hexdigest() != descriptors[key]["sha256"]:
                        raise ValueError(f"Report data checksum mismatch: {key}")
                    seen.add(key)
                    yield key, source
    if fragments or seen != set(descriptors):
        raise ValueError("Incomplete report data bundles")


def read_bundle_value(root, key, descriptor):
    return json.loads(next(source for _, source in iter_bundle_sources(root, {key: descriptor})))


class CSVParts:
    """Split only between rows, repeating the UTF-8 BOM and header in each part."""
    def __init__(self, root, stem, fieldnames, part_bytes=PART_BYTES):
        self.root = Path(root)
        self.stem = stem
        self.fieldnames = fieldnames
        self.part_bytes = part_bytes
        self.parts = []
        self.file = None
        self.size = 0
        self.header = b"\xef\xbb\xbf" + self._encode(dict(zip(fieldnames, fieldnames)))

    def _encode(self, row):
        buffer = io.StringIO(newline="")
        csv.DictWriter(buffer, fieldnames=self.fieldnames, extrasaction="ignore").writerow(row)
        return buffer.getvalue().encode("utf-8")

    def add(self, row):
        encoded = self._encode(row)
        if len(self.header) + len(encoded) > self.part_bytes:
            raise ValueError(f"A single {self.stem} CSV row exceeds the file-size limit")
        if self.file is None or self.size + len(encoded) > self.part_bytes:
            self._next_file()
        self.file.write(encoded)
        self.size += len(encoded)

    def _next_file(self):
        self.close(ensure_header=False)
        number = len(self.parts) + 1
        relative = f"{self.stem}.csv" if number == 1 else f"{self.stem}-part-{number:04d}.csv"
        self.parts.append(relative)
        self.file = (self.root / relative).open("wb")
        self.file.write(self.header)
        self.size = len(self.header)

    def close(self, ensure_header=True):
        if ensure_header and not self.parts:
            self._next_file()
        if self.file is not None:
            self.file.close()
            self.file = None


def write_json_parts(root, stem, rows, part_bytes=PART_BYTES):
    parts = []
    file = None
    size = 0
    for row in rows:
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if len(encoded) + 2 > part_bytes:
            raise ValueError(f"A single {stem} JSON entry exceeds the file-size limit")
        if file is None or size + len(encoded) + 2 > part_bytes:
            if file is not None:
                file.write(b"]"); file.close()
            relative = f"{stem}-part-{len(parts)+1:04d}.json"
            parts.append(relative)
            file = (Path(root) / relative).open("wb")
            file.write(b"["); size = 1
        if size > 1:
            file.write(b","); size += 1
        file.write(encoded); size += len(encoded)
    if file is not None:
        file.write(b"]"); file.close()
    return parts


def validate_report_tree(directory, max_files=MAX_REPORT_FILES):
    files = [path for path in Path(directory).rglob("*") if path.is_file()]
    too_large = [str(path) for path in files if path.stat().st_size > MAX_FILE_BYTES]
    if too_large:
        raise ValueError("Report files exceed 25 MB: " + ", ".join(too_large))
    if len(files) > max_files:
        raise ValueError(f"Report has {len(files)} files; limit is {max_files}")
    return dict(files=len(files), largest_bytes=max((path.stat().st_size for path in files), default=0))


def validate_combined_reports(root, replacements):
    """Apply the file budget to all report languages and all report directories."""
    excluded = [Path(path).resolve() for pair in replacements.items() for path in pair]
    count = 0
    for path in Path(root).rglob("*"):
        if not path.is_file() or any(parent == path.resolve() or parent in path.resolve().parents for parent in excluded):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Report file exceeds 25 MB: {path}")
        count += 1
    for staged in replacements.values():
        count += validate_report_tree(staged)["files"]
    if count > MAX_REPORT_FILES:
        raise ValueError(f"Combined reports would have {count} files; limit is {MAX_REPORT_FILES}")
    return count


def publish_report(staged, target):
    """Replace a checked generated report, restoring the original on failure."""
    staged, target = Path(staged), Path(target)
    backup = None
    if target.exists():
        if not (target / "catalog.json").exists():
            raise ValueError(f"Refusing to replace a directory without a report catalog: {target}")
        backup = Path(tempfile.mkdtemp(prefix=".report-backup-", dir=target.parent))
        backup.rmdir()
        target.rename(backup)
    try:
        staged.rename(target)
    except BaseException:
        if backup is not None:
            backup.rename(target)
        raise
    if backup is not None:
        shutil.rmtree(backup)
