#!/usr/bin/env python3
"""Draw minimal eight-GPU CPU-router packet-flow GIFs for the README."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs/assets"
PREVIEW = ROOT / "build/figures"
W, H = 1600, 900
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
INK = "#17324D"
BLUE = "#0068B5"
TEAL = "#148D91"
ORANGE = "#D65A00"
LIGHT = "#F5F9FC"
LINE = "#B7CAD7"
WHITE = "#FFFFFF"
SHARDS = ("#0068B5", "#D65A00", "#148D91", "#7658A5",
          "#B84B83", "#7F781B", "#247B54", "#996038")
GROUPS = ((50, 770), (830, 1550))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def label(draw: ImageDraw.ImageDraw, x: float, y: float, value: str,
          size: int = 28, color: str = INK, bold: bool = False) -> None:
    face = font(size, bold)
    box = draw.textbbox((0, 0), value, font=face)
    draw.text((x - (box[2] - box[0]) / 2,
               y - (box[3] - box[1]) / 2 - box[1]),
              value, font=face, fill=color)


def rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int],
         fill: str, outline: str = LINE, radius: int = 18,
         width: int = 3) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill,
                           outline=outline, width=width)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float],
          end: tuple[float, float], color: str, t: float,
          width: int = 7) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    wing = 12
    draw.polygon([end,
                  (end[0] - 18 * math.cos(angle) + wing * math.sin(angle),
                   end[1] - 18 * math.sin(angle) - wing * math.cos(angle)),
                  (end[0] - 18 * math.cos(angle) - wing * math.sin(angle),
                   end[1] - 18 * math.sin(angle) + wing * math.cos(angle))],
                 fill=color)
    x = start[0] + (end[0] - start[0]) * t
    y = start[1] + (end[1] - start[1]) * t
    draw.ellipse((x - 11, y - 11, x + 11, y + 11),
                 fill=WHITE, outline=color, width=5)


def chip_x(rank: int) -> int:
    return (86 if rank < 4 else 866) + (rank % 4) * 166


def host_x(rank: int) -> int:
    return (94 if rank < 4 else 874) + (rank % 4) * 160


def base(draw: ImageDraw.ImageDraw, title: str, phases: tuple[str, ...],
         stage: int) -> None:
    label(draw, 800, 48, title, 46, INK, True)
    for index, phase in enumerate(phases):
        left = 95 + index * 355
        active = index == stage
        rect(draw, (left, 100, left + 335, 154),
             BLUE if active else WHITE,
             BLUE if active else LINE, 16, 3)
        label(draw, left + 167, 127, phase, 25,
              WHITE if active else INK, active)
    for node, (left, right) in enumerate(GROUPS):
        rect(draw, (left, 178, right, 858), LIGHT, LINE, 26, 3)
        label(draw, (left + right) / 2, 203, f"NUMA {node}", 27, INK, True)
        rect(draw, (left + 24, 476, right - 24, 607), WHITE, LINE, 20, 3)
        label(draw, left + 125, 502, f"HOST {node}", 23, INK, True)
        rect(draw, (left + 100, 689, right - 100, 788),
             "#E7F5F3", TEAL, 20, 3)


def host_slots(draw: ImageDraw.ImageDraw, operation: str,
               stage: int) -> None:
    for rank in range(8):
        x = host_x(rank)
        rect(draw, (x, 530, x + 144, 584), "#ECF4F9", LINE, 10, 2)
        if operation == "allreduce":
            if stage <= 1:
                draw.ellipse((x + 55, 539, x + 89, 573), fill=SHARDS[rank])
            else:
                label(draw, x + 72, 557, "Σ8", 25, TEAL, True)
        elif operation == "allgather":
            draw.rectangle((x + 49, 539, x + 95, 575), fill=SHARDS[rank])
        else:
            if stage <= 1:
                draw.rectangle((x + 45, 539, x + 99, 575), fill=BLUE)
                label(draw, x + 72, 557, "7", 24, WHITE, True)
            else:
                label(draw, x + 72, 557, "Σ7", 25, TEAL, True)


def gpu_cards(draw: ImageDraw.ImageDraw, operation: str, stage: int) -> None:
    for rank in range(8):
        x = chip_x(rank)
        rect(draw, (x, 242, x + 152, 369), WHITE, BLUE, 19, 4)
        rect(draw, (x + 14, 254, x + 138, 299), BLUE, BLUE, 10, 1)
        label(draw, x + 76, 276, f"G{rank}", 30, WHITE, True)
        if operation == "allreduce":
            if stage < 3:
                draw.ellipse((x + 59, 316, x + 93, 350), fill=SHARDS[rank])
            else:
                label(draw, x + 76, 334, "Σ8", 30, TEAL, True)
        elif operation == "allgather":
            if stage < 3:
                draw.rectangle((x + 58, 316, x + 94, 350),
                               fill=SHARDS[rank])
            else:
                for shard in range(8):
                    x0 = x + 11 + shard * 17
                    draw.rectangle((x0, 319, x0 + 14, 348),
                                   fill=SHARDS[shard])
        elif stage < 3:
            draw.rectangle((x + 25, 317, x + 63, 351), fill=ORANGE)
            draw.rectangle((x + 87, 317, x + 125, 351), fill=BLUE)
            label(draw, x + 106, 334, "7", 22, WHITE, True)
        else:
            draw.rectangle((x + 39, 316, x + 73, 350), fill=ORANGE)
            label(draw, x + 88, 333, "+", 29, INK, True)
            draw.rectangle((x + 106, 316, x + 139, 350), fill=TEAL)


def upload(draw: ImageDraw.ImageDraw, t: float) -> None:
    for rank in range(8):
        x = chip_x(rank) + 76
        arrow(draw, (x, 377), (x, 467), SHARDS[rank], t, 5)


def read_group(draw: ImageDraw.ImageDraw, t: float) -> None:
    for rank in range(8):
        x = chip_x(rank) + 76
        arrow(draw, (x, 467), (x, 379), BLUE, t, 5)


def allgather_read(draw: ImageDraw.ImageDraw, t: float) -> None:
    # Each host bank fans out to all destination GPUs. The diagonal arrows
    # carry the opposite NUMA group's shards into the local fan-out lane.
    for node in range(2):
        first = node * 4
        x0 = chip_x(first) + 76
        x1 = chip_x(first + 3) + 76
        mid = (x0 + x1) / 2
        draw.line(((x0, 420), (x1, 420)), fill=BLUE, width=6)
        arrow(draw, (mid, 468), (mid, 426), BLUE, t, 6)
        for rank in range(first, first + 4):
            x = chip_x(rank) + 76
            arrow(draw, (x, 415), (x, 379), BLUE, t, 5)
    arrow(draw, (410, 466), (1190, 426), ORANGE, t, 6)
    arrow(draw, (1190, 466), (410, 426), ORANGE, t, 6)


def cpu_arrows(draw: ImageDraw.ImageDraw, t: float) -> None:
    for x in (410, 1190):
        arrow(draw, (x, 614), (x, 681), TEAL, t)


def merge_arrows(draw: ImageDraw.ImageDraw, t: float) -> None:
    arrow(draw, (676, 726), (922, 726), ORANGE, t)
    arrow(draw, (924, 750), (678, 750), ORANGE, t)
    for x in (410, 1190):
        arrow(draw, (x, 682), (x, 615), TEAL, t)


def scene(operation: str, stage: int, t: float) -> Image.Image:
    titles = {"allreduce": "AllReduce", "allgather": "AllGather",
              "reducescatter": "ReduceScatter"}
    phases = {"allreduce": ("UPLOAD", "REDUCE", "MERGE", "READ"),
              "allgather": ("UPLOAD", "READY", "READ", "OUTPUT"),
              "reducescatter": ("UPLOAD 7/8", "REDUCE", "MERGE", "ADD OWN")}
    image = Image.new("RGB", (W, H), WHITE)
    draw = ImageDraw.Draw(image)
    base(draw, titles[operation], phases[operation], stage)
    host_slots(draw, operation, stage)
    for node, (left, right) in enumerate(GROUPS):
        mid = (left + right) // 2
        if operation == "allgather":
            label(draw, mid, 737, "✓" if stage >= 1 else "·", 62,
                  TEAL if stage >= 1 else LINE, True)
        elif operation == "allreduce":
            label(draw, mid, 737, "Σ4" if stage <= 1 else "Σ8", 53,
                  TEAL, True)
        else:
            label(draw, mid, 737, "Σ", 60, TEAL, True)
    if stage == 0:
        upload(draw, t)
    elif operation == "allreduce":
        if stage == 1:
            cpu_arrows(draw, t)
        elif stage == 2:
            merge_arrows(draw, t)
        else:
            read_group(draw, t)
    elif operation == "allgather":
        if stage == 1:
            for x in (410, 1190):
                arrow(draw, (x, 615), (x, 680), TEAL, t)
        elif stage == 2:
            allgather_read(draw, t)
    else:
        if stage == 1:
            cpu_arrows(draw, t)
        elif stage == 2:
            merge_arrows(draw, t)
        else:
            read_group(draw, t)
    gpu_cards(draw, operation, stage)
    return image


def render(operation: str) -> None:
    frames, durations = [], []
    for stage in range(4):
        for t in (0.12, 0.40, 0.70, 0.96):
            frame = scene(operation, stage, t)
            frames.append(frame.quantize(colors=128,
                                         method=Image.Quantize.MEDIANCUT))
            durations.append(470 if t == 0.96 else 150)
    durations[-1] = 900
    output = ASSETS / f"cpu-router-8l4-{operation}.gif"
    frames[0].save(output, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=2, optimize=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    for stage in range(4):
        scene(operation, stage, 0.70).save(
            PREVIEW / f"cpu-router-8l4-{operation}-stage{stage}.png")
    print(output.relative_to(ROOT), output.stat().st_size, "bytes")


def main() -> None:
    for operation in ("allreduce", "allgather", "reducescatter"):
        render(operation)


if __name__ == "__main__":
    main()
