#!/usr/bin/env python3
"""Compute linear-HDR render regression metrics from RGB PFM files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
from typing import Dict, Iterable, List, Sequence, Tuple


SIGNALS = ("raw", "diffuse", "specular", "final")

# Normalized top-left image-space ROIs. They intentionally cover silhouettes
# rather than one exact pixel line so they remain useful if resolution changes.
ROI_NORMALIZED = {
    "full_frame": (0.0, 0.0, 1.0, 1.0),
    "ceiling_light_border": (0.38, 0.07, 0.62, 0.21),
    "short_box_silhouette": (0.19, 0.60, 0.55, 0.96),
    "tall_box_silhouette": (0.44, 0.39, 0.77, 0.91),
}


class PFMImage:
    def __init__(self, width: int, height: int, pixels: List[float]):
        self.width = width
        self.height = height
        self.pixels = pixels


def _read_non_comment_line(stream) -> bytes:
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("unexpected end of PFM header")
        line = line.partition(b"#")[0].strip()
        if line:
            return line


def read_pfm(path: Path) -> PFMImage:
    with path.open("rb") as stream:
        magic = _read_non_comment_line(stream)
        if magic != b"PF":
            raise ValueError(f"{path}: expected RGB PFM magic PF")

        dims = _read_non_comment_line(stream).split()
        if len(dims) != 2:
            raise ValueError(f"{path}: invalid PFM dimensions")

        width, height = map(int, dims)
        if width <= 0 or height <= 0:
            raise ValueError(f"{path}: non-positive PFM dimensions")

        scale = float(_read_non_comment_line(stream))
        if scale == 0.0:
            raise ValueError(f"{path}: PFM scale must be non-zero")

        endian = "<" if scale < 0.0 else ">"
        count = width * height * 3
        payload = stream.read()
        expected_bytes = count * 4
        if len(payload) != expected_bytes:
            raise ValueError(
                f"{path}: expected {expected_bytes} payload bytes, "
                f"found {len(payload)}"
            )

        values = list(struct.unpack(f"{endian}{count}f", payload))

    # C++ writes framebuffer rows bottom-to-top. Convert to conventional
    # top-left row-major indexing used by PNGs and ROI definitions.
    row_stride = width * 3
    top_down = [0.0] * count
    for output_y in range(height):
        source_y = height - 1 - output_y
        src = source_y * row_stride
        dst = output_y * row_stride
        top_down[dst : dst + row_stride] = values[src : src + row_stride]

    for value in top_down:
        if not math.isfinite(value):
            raise ValueError(f"{path}: non-finite linear HDR value")

    return PFMImage(width, height, top_down)


def aces_film(value: float) -> float:
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    denominator = value * (c * value + d) + e
    if abs(denominator) <= 1e-20:
        mapped = 0.0
    else:
        mapped = (value * (a * value + b)) / denominator
    return min(max(mapped, 0.0), 1.0)


def display_transform(value: float) -> float:
    return aces_film(value) ** (1.0 / 2.2)


def roi_bounds(
    width: int,
    height: int,
    normalized: Sequence[float],
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = normalized
    ix0 = max(0, min(width, int(math.floor(x0 * width))))
    iy0 = max(0, min(height, int(math.floor(y0 * height))))
    ix1 = max(ix0 + 1, min(width, int(math.ceil(x1 * width))))
    iy1 = max(iy0 + 1, min(height, int(math.ceil(y1 * height))))
    return ix0, iy0, ix1, iy1


def iter_roi_channels(
    candidate: PFMImage,
    reference: PFMImage,
    bounds: Tuple[int, int, int, int],
) -> Iterable[Tuple[float, float]]:
    if (
        candidate.width != reference.width
        or candidate.height != reference.height
    ):
        raise ValueError("candidate/reference dimensions do not match")

    x0, y0, x1, y1 = bounds
    width = candidate.width

    for y in range(y0, y1):
        for x in range(x0, x1):
            base = (y * width + x) * 3
            for channel in range(3):
                yield (
                    candidate.pixels[base + channel],
                    reference.pixels[base + channel],
                )


def compute_metrics(
    candidate: PFMImage,
    reference: PFMImage,
    bounds: Tuple[int, int, int, int],
) -> Dict[str, float | int | None]:
    count = 0
    squared_error = 0.0
    absolute_error = 0.0
    max_absolute_error = 0.0
    reference_energy = 0.0
    display_squared_error = 0.0

    for candidate_value, reference_value in iter_roi_channels(
        candidate, reference, bounds
    ):
        delta = candidate_value - reference_value
        abs_delta = abs(delta)

        squared_error += delta * delta
        absolute_error += abs_delta
        max_absolute_error = max(max_absolute_error, abs_delta)
        reference_energy += reference_value * reference_value

        display_delta = (
            display_transform(candidate_value)
            - display_transform(reference_value)
        )
        display_squared_error += display_delta * display_delta
        count += 1

    if count == 0:
        raise ValueError("empty ROI")

    mse = squared_error / count
    rmse = math.sqrt(mse)
    mae = absolute_error / count
    reference_rms = math.sqrt(reference_energy / count)
    nrmse = rmse / max(reference_rms, 1e-12)

    display_mse = display_squared_error / count
    displayed_psnr = (
        None
        if display_mse <= 0.0
        else 10.0 * math.log10(1.0 / display_mse)
    )

    return {
        "samples": count,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_absolute_error": max_absolute_error,
        "reference_rms": reference_rms,
        "nrmse": nrmse,
        "tone_mapped_psnr_db": displayed_psnr,
    }


def signal_path(validation_dir: Path, spp: int, signal: str) -> Path:
    return validation_dir / f"{spp}spp" / f"cornell_cpu_{signal}.pfm"


def metrics_for_pair(
    candidate: PFMImage,
    reference: PFMImage,
) -> Dict[str, Dict[str, float | int | None]]:
    result = {}
    for name, normalized in ROI_NORMALIZED.items():
        bounds = roi_bounds(
            candidate.width,
            candidate.height,
            normalized,
        )
        result[name] = compute_metrics(
            candidate,
            reference,
            bounds,
        )
    return result


def validate_convergence(report: dict, spps: Sequence[int]) -> None:
    if len(spps) < 3:
        return

    low = str(spps[0])
    mid = str(spps[1])

    for signal in ("raw", "final"):
        low_rmse = report["signal_convergence"][low][signal][
            "full_frame"
        ]["rmse"]
        mid_rmse = report["signal_convergence"][mid][signal][
            "full_frame"
        ]["rmse"]

        if not mid_rmse < low_rmse:
            raise SystemExit(
                f"{signal} full-frame RMSE did not improve: "
                f"{spps[0]} SPP={low_rmse:.8g}, "
                f"{spps[1]} SPP={mid_rmse:.8g}"
            )


def validate_denoising_improvement(report: dict, spps: Sequence[int]) -> None:
    reference_spp = int(report["reference"]["spp"])

    for spp in spps:
        if spp >= reference_spp:
            continue

        key = str(spp)
        for roi in ROI_NORMALIZED:
            raw = report["quality_to_raw_reference"][key]["raw"][roi]
            final = report["quality_to_raw_reference"][key]["final"][roi]

            if not final["rmse"] < raw["rmse"]:
                raise SystemExit(
                    f"{spp} SPP {roi}: denoised Final RMSE did not improve "
                    f"over Raw ({final['rmse']:.8g} >= {raw['rmse']:.8g})"
                )

            final_psnr = final["tone_mapped_psnr_db"]
            raw_psnr = raw["tone_mapped_psnr_db"]
            if (
                final_psnr is not None
                and raw_psnr is not None
                and not final_psnr > raw_psnr
            ):
                raise SystemExit(
                    f"{spp} SPP {roi}: denoised Final tone-mapped PSNR "
                    f"did not improve over Raw "
                    f"({final_psnr:.6g} <= {raw_psnr:.6g})"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-dir",
        type=Path,
        required=True,
        help="Directory containing <spp>spp/cornell_cpu_*.pfm",
    )
    parser.add_argument(
        "--spp",
        type=int,
        nargs="+",
        default=[16, 64, 256],
        help="SPP levels to compare",
    )
    parser.add_argument(
        "--reference-spp",
        type=int,
        default=256,
        help="Same-run provisional reference SPP",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON report destination",
    )
    parser.add_argument(
        "--check-convergence",
        action="store_true",
        help="Fail if 64 SPP does not improve over 16 SPP",
    )
    parser.add_argument(
        "--check-denoising",
        action="store_true",
        help=(
            "Fail when low-SPP Final is not closer than Raw to the "
            "Linear HDR raw reference in full-frame and ROI metrics"
        ),
    )
    args = parser.parse_args()

    spps = sorted(dict.fromkeys(args.spp))
    if args.reference_spp not in spps:
        raise SystemExit("--reference-spp must be present in --spp")

    images: Dict[int, Dict[str, PFMImage]] = {}
    for spp in spps:
        images[spp] = {}
        for signal in SIGNALS:
            path = signal_path(
                args.validation_dir,
                spp,
                signal,
            )
            images[spp][signal] = read_pfm(path)

    reference_images = images[args.reference_spp]
    raw_reference = reference_images["raw"]

    report = {
        "schema_version": 1,
        "reference": {
            "spp": args.reference_spp,
            "role": (
                "same-run provisional convergence reference; "
                "manual 1024/2048 SPP reference is higher quality"
            ),
        },
        "roi_normalized_top_left": ROI_NORMALIZED,
        "signal_convergence": {},
        "quality_to_raw_reference": {},
    }

    for spp in spps:
        key = str(spp)
        report["signal_convergence"][key] = {}
        for signal in SIGNALS:
            report["signal_convergence"][key][signal] = metrics_for_pair(
                images[spp][signal],
                reference_images[signal],
            )

        report["quality_to_raw_reference"][key] = {
            "raw": metrics_for_pair(
                images[spp]["raw"],
                raw_reference,
            ),
            "final": metrics_for_pair(
                images[spp]["final"],
                raw_reference,
            ),
        }

    if args.check_convergence:
        validate_convergence(report, spps)

    if args.check_denoising:
        validate_denoising_improvement(report, spps)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for spp in spps[:-1]:
        raw = report["quality_to_raw_reference"][str(spp)]["raw"][
            "full_frame"
        ]
        final = report["quality_to_raw_reference"][str(spp)]["final"][
            "full_frame"
        ]
        print(
            f"{spp} SPP vs {args.reference_spp} SPP raw reference: "
            f"raw NRMSE={raw['nrmse']:.6f}, "
            f"final NRMSE={final['nrmse']:.6f}, "
            f"raw RMSE={raw['rmse']:.6f}, "
            f"final RMSE={final['rmse']:.6f}"
        )

    print(args.output)


if __name__ == "__main__":
    main()
