import os
import sys
import math
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import RBFInterpolator
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen

def qimage_to_np(img):
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    ptr = img.bits()
    return np.frombuffer(ptr, np.uint8).reshape((h, w, 4)).copy()

def np_to_qimage(arr):
    arr = np.ascontiguousarray(arr)
    h, w, c = arr.shape
    img = QImage(arr.data, w, h, w * c, QImage.Format.Format_RGBA8888)
    return img.copy()

def refine_clean_sprite(sub_arr):
    """
    Extracts fly sprite from white/near-white backdrop:
    - Eliminates leg glowing/white halos via mathematical defringing and deep outer zone ramping.
    - Preserves interior highlights (eyes, patterns) and thin appendages.
    """
    rgb = sub_arr[:, :, :3].astype(np.float32)
    whiteness = np.min(rgb, axis=-1)

    # 1. Outer background zone: light pixels connected to border
    is_light = (whiteness > 150.0)
    border = np.zeros_like(is_light, dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

    lbl, nlbl = ndi.label(is_light)
    border_lbls = np.unique(lbl[border])
    border_lbls = border_lbls[border_lbls > 0]
    is_outer_bg = np.isin(lbl, border_lbls) | (whiteness > 244.0)

    # 2. Deep alpha ramp:
    # 242 (pure transparent) down to 110 (pure opaque)
    t_high = 242.0
    t_low = 110.0

    alpha = np.ones_like(whiteness, dtype=np.float32) * 255.0
    ramp = np.clip((t_high - whiteness) / (t_high - t_low), 0.0, 1.0)
    alpha[is_outer_bg] = ramp[is_outer_bg] * 255.0
    alpha[whiteness >= 242.0] = 0.0

    # 3. Component filtering: remove tiny detached JPEG noise specks
    fg = (alpha > 15.0)
    f_lbl, f_num = ndi.label(fg)
    if f_num > 0:
        sizes = ndi.sum(fg, f_lbl, range(1, f_num + 1))
        max_size = sizes.max()
        # Keep components >= 2% of the main body (keeps legs, antennae, wings)
        valid_lbls = np.where(sizes >= 0.02 * max_size)[0] + 1
        valid_mask = np.isin(f_lbl, valid_lbls)
        alpha[~valid_mask] = 0.0

    # 4. Defringing: mathematically unmix white paper background from RGB
    # C = a * F + (1 - a) * 255  ==>  F = (C - 255 * (1 - a)) / a
    a_norm = np.clip(alpha / 255.0, 0.05, 1.0)[:, :, np.newaxis]
    rgb_clean = np.clip((rgb - 255.0 * (1.0 - a_norm)) / a_norm, 0.0, 255.0)

    out = np.zeros((sub_arr.shape[0], sub_arr.shape[1], 4), dtype=np.uint8)
    out[:, :, :3] = np.where(a_norm > 0.02, rgb_clean, rgb).astype(np.uint8)
    out[:, :, 3] = alpha.astype(np.uint8)
    return out

def process_sprite(sheet_arr, crop_box, target_size=128, rot_k=-1):
    """Crops, cleans alpha, rotates (k=-1 for 90 CW, k=1 for 90 CCW), and centers on canvas."""
    x0, y0, x1, y1 = crop_box
    sub = sheet_arr[y0:y1, x0:x1]
    if sub.size == 0:
        return np.zeros((target_size, target_size, 4), dtype=np.uint8)

    cleaned = refine_clean_sprite(sub)
    rot = np.ascontiguousarray(np.rot90(cleaned, k=rot_k))

    alpha = rot[:, :, 3]
    y_idx, x_idx = np.where(alpha > 10)
    if len(y_idx) == 0:
        return np.zeros((target_size, target_size, 4), dtype=np.uint8)

    cy0, cy1 = y_idx.min(), y_idx.max()
    cx0, cx1 = x_idx.min(), x_idx.max()
    content = rot[cy0:cy1+1, cx0:cx1+1]
    ch, cw = content.shape[:2]

    # Scale to fit canvas leaving a clean margin
    margin = 8
    avail = target_size - 2 * margin
    scale = min(1.0, avail / max(cw, ch))

    q_content = np_to_qimage(content)
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))
    q_scaled = q_content.scaled(new_w, new_h)
    scaled_arr = qimage_to_np(q_scaled)

    canvas = np.zeros((target_size, target_size, 4), dtype=np.uint8)
    off_x = (target_size - new_w) // 2
    off_y = (target_size - new_h) // 2
    canvas[off_y:off_y+new_h, off_x:off_x+new_w] = scaled_arr
    return canvas

def generate_tripod_walk_frames(base_arr, pinned, leg_tips, stride=8.0, lift=2.5, num_frames=8):
    """
    Generates natural alternating tripod gait walking frames from a base sprite.
    Pins the body (head, eyes, thorax, abdomen, folded wings) and canvas borders so they remain stationary.
    Displaces the 6 leg tips in alternating tripod strides (FL, HL, MR vs FR, ML, HR) using thin plate spline RBF.
    """
    # Alternating tripod gait: Tripod 1 (0 rad) vs Tripod 2 (pi rad)
    # Leg tips order: FL, ML, HL, FR, MR, HR
    leg_phases = [0.0, math.pi, 0.0, math.pi, 0.0, math.pi]

    grid_y, grid_x = np.mgrid[0:128, 0:128]
    grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    walk_frames = []
    for f in range(num_frames):
        phase = (f / float(num_frames)) * math.tau
        src_pts = np.array(pinned + leg_tips, dtype=np.float32)
        target_disp = np.zeros_like(src_pts)

        for i, (lx, ly) in enumerate(leg_tips):
            l_phase = phase + leg_phases[i]
            dx = math.sin(l_phase) * stride
            dy = -abs(math.sin(l_phase)) * lift * (1.0 if ly > 64 else -1.0)
            target_disp[len(pinned) + i] = [dx, dy]

        rbf = RBFInterpolator(src_pts, target_disp, kernel='thin_plate_spline')
        disp = rbf(grid_coords).reshape((128, 128, 2))
        disp_x = disp[:, :, 0]
        disp_y = disp[:, :, 1]

        map_x = np.clip(grid_x - disp_x, 0, 127)
        map_y = np.clip(grid_y - disp_y, 0, 127)

        warped = np.zeros_like(base_arr)
        for c in range(4):
            warped[:, :, c] = ndi.map_coordinates(base_arr[:, :, c], [map_y, map_x], order=1, mode='constant', cval=0)
        walk_frames.append(warped)

    return walk_frames

def generate_thumbnails(skins_root):
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication(sys.argv)
    from neuropest.render import AVAILABLE_SKINS, draw_fly
    thumb_dir = os.path.join(skins_root, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)
    size = 64
    for key in AVAILABLE_SKINS.keys():
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # Subtle dark badge background (rounded circle)
        p.setPen(QPen(QColor(51, 65, 85), 1.5))
        p.setBrush(QColor(15, 23, 42, 240))
        p.drawEllipse(2, 2, size - 4, size - 4)

        # Fly centered at (32, 32), angled naturally towards upper-right
        draw_fly(p, size / 2.0, size / 2.0, heading=-math.pi / 4.0, state="stand", phase=0.0,
                 scale=1.05, feed_glow=0.0, skin=key)

        p.end()
        out_file = os.path.join(thumb_dir, f"{key}.png")
        img.save(out_file)
        print(f"  Saved thumbnail: {out_file}")


def main():
    skins_root = "neuropest/assets/skins"
    os.makedirs(skins_root, exist_ok=True)

    # Canvas border pins shared by all skins
    canvas_pins = [
        [0, 0], [127, 0], [0, 127], [127, 127],
        [64, 0], [64, 127], [0, 64], [127, 64],
    ]

    # ---------------- 1. CHUBBY (Tombul Chibi) ----------------
    print("Processing Chubby...")
    chubby_dir = os.path.join(skins_root, "chubby")
    os.makedirs(chubby_dir, exist_ok=True)

    arr_cwalk = qimage_to_np(QImage("assets/fly_skins/07_chubby_fly_topdown_walk.jpg"))
    chubby_idle = process_sprite(arr_cwalk, (20, 35, 236, 240))
    np_to_qimage(chubby_idle).save(os.path.join(chubby_dir, "idle.png"))

    chubby_pinned = [
        [108, 59], [108, 69],        # Antennae
        [100, 42], [100, 86], [90, 64], # Eyes & head
        [68, 64],                    # Thorax
        [45, 64], [30, 64],          # Abdomen
        [40, 38], [40, 90],          # Folded wings
    ] + canvas_pins
    chubby_leg_tips = [
        [100, 20],   # FL
        [52, 10],    # ML
        [14, 20],    # HL
        [100, 108],  # FR
        [52, 118],   # MR
        [14, 108],   # HR
    ]
    chubby_walk = generate_tripod_walk_frames(chubby_idle, chubby_pinned, chubby_leg_tips, stride=8.0, lift=2.5)
    for i, spr in enumerate(chubby_walk):
        np_to_qimage(spr).save(os.path.join(chubby_dir, f"walk_{i}.png"))

    arr_cfly = qimage_to_np(QImage("assets/fly_skins/06_chubby_fly_topdown_flight.jpg"))
    cfly_row0 = [(45, 145), (210, 315), (370, 485), (530, 660), (695, 825), (865, 990)]
    cfly_row2 = [(35, 155), (195, 345), (345, 510), (525, 665), (710, 815), (875, 980)]
    f_idx = 0
    for x0, x1 in cfly_row0:
        spr = process_sprite(arr_cfly, (x0, 30, x1, 185))
        np_to_qimage(spr).save(os.path.join(chubby_dir, f"fly_{f_idx}.png"))
        f_idx += 1
    for x0, x1 in cfly_row2:
        spr = process_sprite(arr_cfly, (x0, 285, x1, 440))
        np_to_qimage(spr).save(os.path.join(chubby_dir, f"fly_{f_idx}.png"))
        f_idx += 1

    # ---------------- 2. CYBORG (Siber Mecha) ----------------
    print("Processing Cyborg...")
    cyborg_dir = os.path.join(skins_root, "cyborg")
    os.makedirs(cyborg_dir, exist_ok=True)
    arr_cyborg = qimage_to_np(QImage("assets/fly_skins/09_cyborg_fly_topdown.jpg"))

    # Flight frames: head at top in rows 0 & 1, rotate 90 CW (k=-1) -> head facing +x
    cyb_fly_boxes = [
        (35, 35, 315, 245), (370, 30, 655, 245), (700, 30, 1000, 245),
        (25, 280, 320, 490), (365, 280, 655, 490), (705, 280, 995, 490),
    ]
    for i, box in enumerate(cyb_fly_boxes):
        spr = process_sprite(arr_cyborg, box, rot_k=-1)
        np_to_qimage(spr).save(os.path.join(cyborg_dir, f"fly_{i}.png"))

    # Walk base sprite: head is at bottom in raw image, rotate 90 CCW (k=1) -> head facing +x
    cyb_idle = process_sprite(arr_cyborg, (45, 525, 305, 750), rot_k=1)
    np_to_qimage(cyb_idle).save(os.path.join(cyborg_dir, "idle.png"))

    cyborg_pinned = [
        [98, 64], [92, 64],          # Mandibles & head
        [87, 53], [87, 75],          # Eyes
        [78, 64], [60, 64], [50, 64], # Thorax & core
        [35, 64], [25, 64], [15, 64], # Abdomen
        [47, 50], [60, 50], [72, 53], # Left leg roots (coxa)
        [47, 78], [60, 78], [72, 75], # Right leg roots (coxa)
    ] + canvas_pins
    cyborg_leg_tips = [
        [106, 23],  # FL
        [75, 10],   # ML
        [38, 17],   # HL
        [106, 105], # FR
        [75, 118],  # MR
        [38, 111],  # HR
    ]
    cyb_walk = generate_tripod_walk_frames(cyb_idle, cyborg_pinned, cyborg_leg_tips, stride=8.5, lift=2.5)
    for i, spr in enumerate(cyb_walk):
        np_to_qimage(spr).save(os.path.join(cyborg_dir, f"walk_{i}.png"))

    # ---------------- 3. CANDY (Pastel Şeker / Peri) ----------------
    print("Processing Candy...")
    candy_dir = os.path.join(skins_root, "candy")
    os.makedirs(candy_dir, exist_ok=True)
    arr_candy = qimage_to_np(QImage("assets/fly_skins/08_pastel_candy_fly_topdown.jpg"))

    candy_fly_boxes = [
        (115, 50, 255, 245), (370, 50, 655, 245), (690, 50, 990, 245),
        (45, 285, 330, 480), (420, 285, 605, 480), (770, 285, 915, 480),
    ]
    for i, box in enumerate(candy_fly_boxes):
        spr = process_sprite(arr_candy, box)
        np_to_qimage(spr).save(os.path.join(candy_dir, f"fly_{i}.png"))

    # Walk base sprite: wide box (270, 520, 500, 760) preserving all 6 legs intact
    candy_idle = process_sprite(arr_candy, (270, 520, 500, 760))
    np_to_qimage(candy_idle).save(os.path.join(candy_dir, "idle.png"))

    candy_pinned = [
        [115, 41], [115, 87],        # Antennae
        [95, 44], [95, 84], [80, 64], # Eyes & head
        [60, 64], [55, 52], [55, 76], # Thorax
        [35, 64], [20, 64],          # Abdomen
        [30, 40], [30, 88],          # Folded wings
    ] + canvas_pins
    candy_leg_tips = [
        [97, 26],   # FL
        [72, 13],   # ML
        [8, 30],    # HL
        [97, 98],   # FR
        [62, 114],  # MR
        [8, 86],    # HR
    ]
    candy_walk = generate_tripod_walk_frames(candy_idle, candy_pinned, candy_leg_tips, stride=8.0, lift=2.0)
    for i, spr in enumerate(candy_walk):
        np_to_qimage(spr).save(os.path.join(candy_dir, f"walk_{i}.png"))

    # ---------------- 4. CARTOON (Çizgi Film) ----------------
    print("Processing Cartoon...")
    cartoon_dir = os.path.join(skins_root, "cartoon")
    os.makedirs(cartoon_dir, exist_ok=True)
    arr_cart = qimage_to_np(QImage("assets/fly_skins/05_cartoon_fly_topdown_dorsal.jpg"))

    cart_fly_boxes = [
        (55, 28, 200, 220), (310, 28, 455, 220), (515, 28, 765, 220), (770, 28, 1020, 220),
        (55, 280, 200, 475), (260, 280, 510, 475), (515, 280, 765, 475), (770, 280, 1020, 475),
    ]
    for i, box in enumerate(cart_fly_boxes):
        spr = process_sprite(arr_cart, box)
        np_to_qimage(spr).save(os.path.join(cartoon_dir, f"fly_{i}.png"))

    cart_idle = process_sprite(arr_cart, (45, 530, 210, 750))
    np_to_qimage(cart_idle).save(os.path.join(cartoon_dir, "idle.png"))

    cartoon_pinned = [
        [105, 64], [95, 52], [95, 76], # Head & eyes
        [70, 64], [70, 52], [70, 76], # Thorax
        [35, 64], [25, 64], [45, 64], # Abdomen
        [30, 42], [50, 38], [30, 86], [50, 90], # Folded wings
        [18, 48], [18, 80],          # Wing tips
    ] + canvas_pins
    cartoon_leg_tips = [
        [106, 32],   # FL
        [65, 23],    # ML
        [14, 38],    # HL
        [106, 96],   # FR
        [65, 105],   # MR
        [14, 90],    # HR
    ]
    cart_walk = generate_tripod_walk_frames(cart_idle, cartoon_pinned, cartoon_leg_tips, stride=7.5, lift=2.0)
    for i, spr in enumerate(cart_walk):
        np_to_qimage(spr).save(os.path.join(cartoon_dir, f"walk_{i}.png"))

    # ---------------- 5. THUMBNAILS ----------------
    print("Generating UI thumbnails...")
    generate_thumbnails(skins_root)

    print("\nAll skins and UI thumbnails successfully generated!")

if __name__ == "__main__":
    main()
