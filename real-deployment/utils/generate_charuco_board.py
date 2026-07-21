"""Generate the fixed A4 ChArUco board used by ViperX hand-eye calibration.

The SVG is the printable artifact.  Print it on A4 paper at 100% / Actual
size with any "Fit to page" option disabled.  The PNG is a board-only preview
for visual inspection and detector checks; it does not define print scale.
"""

import argparse
import base64
from pathlib import Path

import cv2


CHARUCO_BOARD_SHAPE = (5, 5)
CHARUCO_SQUARE_LENGTH_MM = 36.0
CHARUCO_MARKER_LENGTH_MM = 27.0
CHARUCO_BOARD_SIZE_MM = 180.0
CHARUCO_DICTIONARY_NAME = "DICT_6X6_250"

A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0
BOARD_MARGIN_X_MM = (A4_WIDTH_MM - CHARUCO_BOARD_SIZE_MM) / 2.0
BOARD_MARGIN_Y_MM = 15.0
BOARD_IMAGE_PIXELS = 2400

OUTPUT_STEM = "charuco_5x5_180mm_dict_6x6_250"


def create_charuco_board():
    """Return the OpenCV dictionary and the fixed 5x5 ChArUco board."""

    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoBoard"):
        raise RuntimeError(
            "OpenCV ArUco support is required; install an OpenCV contrib build."
        )
    expected_board_size_mm = CHARUCO_BOARD_SHAPE[0] * CHARUCO_SQUARE_LENGTH_MM
    if CHARUCO_BOARD_SHAPE[0] != CHARUCO_BOARD_SHAPE[1]:
        raise ValueError("The printable ChArUco board must be square.")
    if expected_board_size_mm != CHARUCO_BOARD_SIZE_MM:
        raise ValueError(
            "Board size must equal square count multiplied by square length."
        )
    if not 0.0 < CHARUCO_MARKER_LENGTH_MM < CHARUCO_SQUARE_LENGTH_MM:
        raise ValueError("Marker length must be positive and smaller than a square.")

    dictionary_id = getattr(cv2.aruco, CHARUCO_DICTIONARY_NAME)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    board = cv2.aruco.CharucoBoard(
        CHARUCO_BOARD_SHAPE,
        CHARUCO_SQUARE_LENGTH_MM / 1000.0,
        CHARUCO_MARKER_LENGTH_MM / 1000.0,
        dictionary,
    )
    return dictionary, board


def render_board_image(board):
    """Render a high-resolution, board-only grayscale image."""

    return board.generateImage(
        (BOARD_IMAGE_PIXELS, BOARD_IMAGE_PIXELS),
        marginSize=0,
        borderBits=1,
    )


def encode_png(image) -> bytes:
    """Encode an OpenCV image as PNG bytes, failing clearly on encoder errors."""

    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("OpenCV failed to encode the ChArUco PNG.")
    return encoded.tobytes()


def build_a4_svg(png_bytes: bytes) -> str:
    """Embed the board PNG in an exact-size A4 SVG with a 100 mm check line."""

    encoded_png = base64.b64encode(png_bytes).decode("ascii")
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{A4_WIDTH_MM:g}mm" height="{A4_HEIGHT_MM:g}mm"
     viewBox="0 0 {A4_WIDTH_MM:g} {A4_HEIGHT_MM:g}">
  <rect x="0" y="0" width="{A4_WIDTH_MM:g}" height="{A4_HEIGHT_MM:g}" fill="white"/>
  <image x="{BOARD_MARGIN_X_MM:g}" y="{BOARD_MARGIN_Y_MM:g}"
         width="{CHARUCO_BOARD_SIZE_MM:g}" height="{CHARUCO_BOARD_SIZE_MM:g}"
         preserveAspectRatio="none"
         href="data:image/png;base64,{encoded_png}"/>
  <text x="105" y="205" text-anchor="middle" font-family="sans-serif" font-size="4">
    ChArUco 5x5 | DICT_6X6_250 | square 36 mm | marker 27 mm
  </text>
  <line x1="55" y1="220" x2="155" y2="220" stroke="black" stroke-width="0.4"/>
  <line x1="55" y1="217" x2="55" y2="223" stroke="black" stroke-width="0.4"/>
  <line x1="155" y1="217" x2="155" y2="223" stroke="black" stroke-width="0.4"/>
  <text x="105" y="229" text-anchor="middle" font-family="sans-serif" font-size="4">
    This line must measure exactly 100 mm after printing.
  </text>
  <text x="105" y="238" text-anchor="middle" font-family="sans-serif" font-size="4">
    Print at 100% / Actual size. Disable Fit to page.
  </text>
</svg>
'''


def write_board_files(output_dir: Path) -> tuple[Path, Path]:
    """Write the board-only PNG and printable A4 SVG into ``output_dir``."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, board = create_charuco_board()
    board_image = render_board_image(board)
    png_bytes = encode_png(board_image)

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    svg_path = output_dir / f"{OUTPUT_STEM}_a4.svg"
    png_path.write_bytes(png_bytes)
    svg_path.write_text(build_a4_svg(png_bytes), encoding="utf-8")
    return png_path, svg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the fixed 180 mm ViperX hand-eye ChArUco board."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for the board-only PNG and printable A4 SVG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    png_path, svg_path = write_board_files(args.output_dir)
    print(f"board_shape={CHARUCO_BOARD_SHAPE}")
    print(f"board_size_mm={CHARUCO_BOARD_SIZE_MM}")
    print(f"square_length_mm={CHARUCO_SQUARE_LENGTH_MM}")
    print(f"marker_length_mm={CHARUCO_MARKER_LENGTH_MM}")
    print(f"preview_png={png_path}")
    print(f"printable_a4_svg={svg_path}")
    print("print_scale=100% / Actual size; Fit to page disabled")


if __name__ == "__main__":
    main()
