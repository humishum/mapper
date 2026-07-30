"""RFC 7233 single-range parsing and bounded file streaming."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$", re.IGNORECASE)
STREAM_CHUNK_SIZE = 1024 * 1024


class InvalidRange(ValueError):
    pass


@dataclass(frozen=True)
class ByteRange:
    start: int
    end: int
    total_size: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def content_range(self) -> str:
        return f"bytes {self.start}-{self.end}/{self.total_size}"


def parse_range_header(value: str, total_size: int) -> ByteRange:
    if total_size <= 0:
        raise InvalidRange("ranges are not satisfiable for an empty file")
    match = RANGE_PATTERN.fullmatch(value.strip())
    if match is None:
        raise InvalidRange("only one bytes range is supported")
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        raise InvalidRange("range is empty")
    if not start_text:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise InvalidRange("suffix length must be positive")
        start = max(0, total_size - suffix_length)
        end = total_size - 1
    else:
        start = int(start_text)
        if start >= total_size:
            raise InvalidRange("range starts beyond the end of the file")
        end = int(end_text) if end_text else total_size - 1
        if end < start:
            raise InvalidRange("range end precedes range start")
        end = min(end, total_size - 1)
    return ByteRange(start=start, end=end, total_size=total_size)


async def iter_file(path: Path, *, start: int, length: int) -> AsyncIterator[bytes]:
    """Yield only the requested region and observe cancellation between chunks."""
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            block = handle.read(min(STREAM_CHUNK_SIZE, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block
            await asyncio.sleep(0)
