#!/usr/bin/env python3
"""Convert the renderers' ASCII PPM (P3) output to RGB PNG without dependencies."""

import argparse
from pathlib import Path
import struct
import zlib


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def ppm_tokens(source):
    for line in source:
        for token in line.partition(b"#")[0].split():
            yield token


def png_chunk(kind, payload):
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def convert(source_path, output_path):
    with source_path.open("rb") as source:
        tokens = ppm_tokens(source)
        if next(tokens, None) != b"P3":
            raise ValueError(f"{source_path}: expected P3 PPM")

        try:
            width = int(next(tokens))
            height = int(next(tokens))
            max_value = int(next(tokens))
        except (StopIteration, ValueError) as error:
            raise ValueError(f"{source_path}: invalid PPM header") from error
        if width <= 0 or height <= 0 or max_value != 255:
            raise ValueError(f"{source_path}: expected positive dimensions and max value 255")

        scanlines = bytearray()
        for row in range(height):
            scanlines.append(0)  # PNG filter: None
            for _ in range(width * 3):
                token = next(tokens, None)
                if token is None:
                    raise ValueError(f"{source_path}: pixel data ends at row {row}")
                try:
                    value = int(token)
                except ValueError as error:
                    raise ValueError(f"{source_path}: invalid pixel value {token!r}") from error
                if not 0 <= value <= 255:
                    raise ValueError(f"{source_path}: pixel value outside 0..255")
                scanlines.append(value)

        if next(tokens, None) is not None:
            raise ValueError(f"{source_path}: unexpected data after the image")

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        PNG_SIGNATURE
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(scanlines, level=6))
        + png_chunk(b"IEND", b"")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(png)
    print(f"{output_path}: {width}x{height}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="P3 PPM files")
    parser.add_argument("--output-dir", type=Path, help="PNG destination directory")
    args = parser.parse_args()
    for source_path in args.files:
        output_path = (args.output_dir or source_path.parent) / (source_path.stem + ".png")
        convert(source_path, output_path)


if __name__ == "__main__":
    main()
