"""Generate high-resolution multi-size Windows .ico file for NeuroPest."""
import struct
from pathlib import Path
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtWidgets import QApplication
from neuropest.theme import fly_icon


def generate_ico(dest: Path):
    _ = QApplication.instance() or QApplication([])
    sizes = [16, 24, 32, 48, 64, 128, 256]
    png_images = []

    for s in sizes:
        icon = fly_icon(size=s)
        pm = icon.pixmap(s, s)
        buf = QBuffer()
        buf.open(QIODevice.WriteOnly)
        pm.save(buf, "PNG")
        png_data = bytes(buf.data())
        buf.close()
        png_images.append((s, png_data))

    # Build ICO file with embedded PNGs
    count = len(png_images)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count

    dir_entries = []
    data_blobs = []

    for s, data in png_images:
        w = 0 if s == 256 else s
        h = 0 if s == 256 else s
        size_bytes = len(data)
        entry = struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, size_bytes, offset)
        dir_entries.append(entry)
        data_blobs.append(data)
        offset += size_bytes

    ico_bytes = header + b"".join(dir_entries) + b"".join(data_blobs)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(ico_bytes)
    print(f"Generated multi-res icon at {dest} ({len(ico_bytes)} bytes)")


if __name__ == "__main__":
    generate_ico(Path("neuropest.ico"))
