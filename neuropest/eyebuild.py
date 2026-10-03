"""Build-time part of the visual front end: columns, photoreceptor directions and receptive fields from the
connectome (pandas / scipy, needs data/raw). The runtime part is `vision.py`."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .vision import KINDS, VOXEL_UM, Retina  # noqa: F401

# --------------------------------------------------------------------------- building from the connectome
def body_axes(ann, pos):
    """Unit vectors (ahead, right, up) in data coordinates (micrometres) and the head centre.

    right: from the left-labelled to the right-labelled photoreceptors; up: gustatory -> central complex
    and Kenyon cells; ahead: mushroom-body Kenyon cells -> olfactory sensory neurons."""
    def centroid(mask):
        return pos[np.asarray(mask)].mean(0)

    is_pr = (ann.cell_class == "visual") & ann.cell_type.isin(["R1-6", "R7", "R8"])
    left_c, right_c = centroid(is_pr & (ann.side == "left")), centroid(is_pr & (ann.side == "right"))
    right = right_c - left_c
    right /= np.linalg.norm(right)
    up = centroid(ann.cell_class.isin(["CX", "Kenyon_Cell"])) - centroid(ann.cell_class == "gustatory")
    up -= right * (up @ right)
    up /= np.linalg.norm(up)
    ahead = centroid(ann.cell_class == "olfactory") - centroid(ann.cell_class == "Kenyon_Cell")
    ahead -= right * (ahead @ right) + up * (ahead @ up)
    ahead /= np.linalg.norm(ahead)
    return ahead, right, up, (left_c + right_c) / 2


def column_directions(cols_pos: np.ndarray, ahead, right, up):
    """Viewing direction of every medulla column of one eye: from the centre of a sphere fitted to the
    columnar (Mi1) neurons of the eye towards each neuron, in the body frame (azimuth 0 = ahead, positive
    right; elevation positive up). A sphere fits the medulla sheet well (rms about 10% of its radius), and
    neighbours in 3D are neighbours in the visual field. Returns azimuth, elevation (radians), the sphere
    radius and the rms residual (micrometres)."""
    a = np.c_[2 * cols_pos, np.ones(len(cols_pos))]
    x = np.linalg.lstsq(a, (cols_pos ** 2).sum(1), rcond=None)[0]
    c = x[:3]
    radius = float(np.sqrt(x[3] + (c ** 2).sum()))
    d = cols_pos - c
    rms = float(np.sqrt(np.mean((np.linalg.norm(d, axis=1) - radius) ** 2)))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    az = np.arctan2(d @ right, d @ ahead)
    el = np.arcsin(np.clip(d @ up, -1, 1))
    return az.astype(np.float32), el.astype(np.float32), radius, rms


def _strongest(net, row: int, targets) -> int | None:
    """Network index among `targets` (a set or dict keyed by network index) with the strongest synapses
    from neuron `row`, or None."""
    lo, hi = net.indptr[row], net.indptr[row + 1]
    best, best_w = None, 0.0
    for t, w in zip(net.indices[lo:hi], net.data[lo:hi]):
        if int(t) in targets and abs(w) > best_w:
            best, best_w = int(t), abs(w)
    return best


def build_retina(net, annotation_tsv: Path) -> Retina:
    """Photoreceptor viewing directions for the network `net` (needs `net.ids` and the connectivity).

    Each medulla column (one Mi1 neuron) gets a direction (`column_directions`). Photoreceptors reach it
    through the lamina: R1-6 -> strongest lamina monopolar target (L1, L2 or L3) -> the L1 of that cartridge
    (nearest L1 in space) -> its strongest Mi1 target (neural superposition: six photoreceptors from
    neighbouring ommatidia converge on one cartridge and share its view). R7 and R8, and R1-6 without a
    lamina target in this dataset, use the L1 nearest to their own position."""
    import pandas as pd
    from scipy.spatial import cKDTree

    cols = ["root_id", "super_class", "cell_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"]
    ann = pd.read_csv(annotation_tsv, sep="	", usecols=cols, low_memory=False).dropna(subset=["pos_x"])
    pos = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_UM
    ahead, right, up, _ = body_axes(ann, pos)
    row_of = {int(r): i for i, r in enumerate(net.ids)}
    root = ann.root_id.to_numpy()
    net_row = np.array([row_of.get(int(r), -1) for r in root])
    in_net = net_row >= 0
    pos_of_row = {int(r): pos[i] for i, r in enumerate(net_row) if r >= 0}

    idx, az, el, eye, kind = [], [], [], [], []
    for side, side_code in (("left", 0), ("right", 1)):
        def rows_of(cell_type):
            sel = ((ann.cell_type == cell_type) & (ann.side == side)).to_numpy() & in_net
            return sel, net_row[sel]

        mi1_sel, mi1_rows = rows_of("Mi1")
        a_col, e_col, radius, rms = column_directions(pos[mi1_sel], ahead, right, up)
        mi1_col = {int(r): j for j, r in enumerate(mi1_rows)}
        l1_sel, l1_rows = rows_of("L1")
        l1_to_col = np.array([mi1_col.get(_strongest(net, int(r), mi1_col), -1) for r in l1_rows])
        lamina = set()
        for ct in ("L1", "L2", "L3"):
            lamina |= set(int(r) for r in rows_of(ct)[1])
        tree_l1 = cKDTree(pos[l1_sel])
        print(f"  {side}: {len(mi1_rows)} Mi1 columns (sphere radius {radius:.0f} um, rms {rms:.0f} um), "
              f"{len(l1_rows)} L1 cartridges, {int((l1_to_col >= 0).sum())} with an Mi1 partner")
        for ct, kind_code in (("R1-6", 0), ("R7", 1), ("R8", 2)):
            sel = ((ann.cell_class == "visual") & (ann.side == side) & (ann.cell_type == ct)).to_numpy() & in_net
            rows = net_row[sel]
            _, cart = tree_l1.query(pos[sel])                       # default: the L1 nearest to the photoreceptor
            if ct == "R1-6":
                for n, r in enumerate(rows):
                    t = _strongest(net, int(r), lamina)
                    if t is not None:
                        cart[n] = tree_l1.query(pos_of_row[t])[1]
            col = l1_to_col[cart]
            ok = col >= 0
            idx.append(rows[ok].astype(np.int32))
            az.append(a_col[col[ok]])
            el.append(e_col[col[ok]])
            eye.append(np.full(int(ok.sum()), side_code, np.uint8))
            kind.append(np.full(int(ok.sum()), kind_code, np.uint8))
        for ct, kind_code in (("L1", 3), ("L2", 4), ("L3", 5)):     # lamina cells: complete coverage
            sel, rows = rows_of(ct)
            _, cart = tree_l1.query(pos[sel])
            col = l1_to_col[cart]
            ok = col >= 0
            idx.append(rows[ok].astype(np.int32))
            az.append(a_col[col[ok]])
            el.append(e_col[col[ok]])
            eye.append(np.full(int(ok.sum()), side_code, np.uint8))
            kind.append(np.full(int(ok.sum()), kind_code, np.uint8))
    return Retina(np.concatenate(idx), np.concatenate(az), np.concatenate(el), np.concatenate(eye),
                  np.concatenate(kind))




# --------------------------------------------------------------------------- columns and receptive fields
import numba as nb  # noqa: E402


@nb.njit(cache=True)
def _accumulate(indptr, indices, data, known, dirs):
    """acc[post] += |w| * dirs[pre] over the edges whose presynaptic neuron has a known direction."""
    n = indptr.shape[0] - 1
    acc = np.zeros((n, 3), np.float64)
    wsum = np.zeros(n, np.float64)
    for pre in range(n):
        if known[pre]:
            for k in range(indptr[pre], indptr[pre + 1]):
                post = indices[k]
                w = abs(data[k])
                acc[post, 0] += w * dirs[pre, 0]
                acc[post, 1] += w * dirs[pre, 1]
                acc[post, 2] += w * dirs[pre, 2]
                wsum[post] += w
    return acc, wsum


COLUMNAR = r"^(Tm\d+.*|TmY\d+.*|Mi\d+.*|T2.*|T3|Y\d+.*|C2|C3|L4|L5)$"     # one cell per medulla column


def build_visual_field(net, annotation_tsv: Path, conc_min: float = 0.25):
    """Columns (unit direction vectors in the body frame) and receptive field centres of projection neurons.

    Directions: Mi1 columns from a sphere fit (`column_directions`); L1-L3 and the other columnar cells
    (Tm, TmY, Mi, T2, T3, Y, C, L4, L5) take the direction of the nearest Mi1 in space; T4/T5 and then every
    visual projection neuron take the synapse-weighted mean direction of their presynaptic neurons that
    have one, and the length of that mean (0..1) as concentration: near 1 for a small receptive field."""
    import pandas as pd
    from scipy.spatial import cKDTree

    cols = ["root_id", "super_class", "cell_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"]
    ann = pd.read_csv(annotation_tsv, sep="\t", usecols=cols, low_memory=False)
    ann = ann.dropna(subset=["pos_x"]).reset_index(drop=True)
    pos = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * VOXEL_UM
    ahead, right, up, _ = body_axes(ann, pos)
    row_of = {int(r): i for i, r in enumerate(net.ids)}
    net_row = np.array([row_of.get(int(r), -1) for r in ann.root_id])
    ok = net_row >= 0
    dirs = np.full((net.n, 3), np.nan, np.float32)
    known = np.zeros(net.n, bool)
    cell = ann.cell_type.fillna("").to_numpy()
    side = ann.side.fillna("").to_numpy()
    col_dir, col_eye = [], []
    for s, code in (("left", 0), ("right", 1)):
        mi1 = (cell == "Mi1") & (side == s) & ok
        az, el, _, _ = column_directions(pos[mi1], ahead, right, up)
        v = np.c_[np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)].astype(np.float32)
        dirs[net_row[mi1]] = v
        known[net_row[mi1]] = True
        col_dir.append(v)
        col_eye.append(np.full(len(v), code, np.uint8))
        tree = cKDTree(pos[mi1])
        mi1_dir = v
        # columnar cells in this eye (L1-L3 included) take the direction of the nearest Mi1 in space
        is_col = (ann.cell_type.fillna("").str.match(COLUMNAR) | ann.cell_type.isin(["L1", "L2", "L3"])).to_numpy()
        sel = is_col & (side == s) & ok & (cell != "Mi1")
        _, near = tree.query(pos[sel])
        dirs[net_row[sel]] = mi1_dir[near]
        known[net_row[sel]] = True
    # T4/T5 from their columnar inputs, then the projection neurons from everything known so far
    for stage, mask in (("T4T5", ann.cell_type.fillna("").str.match(r"^T[45][a-d]$").to_numpy()),
                        ("VPN", (ann.super_class == "visual_projection").to_numpy())):
        acc, wsum = _accumulate(net.indptr, net.indices, net.data, known, dirs)
        rows = net_row[mask & ok]
        good = wsum[rows] > 0
        mean = acc[rows[good]] / wsum[rows[good]][:, None]
        norm = np.linalg.norm(mean, axis=1)
        if stage == "T4T5":
            dirs[rows[good]] = (mean / norm[:, None]).astype(np.float32)
            known[rows[good]] = True
        else:
            conc = np.zeros(len(rows), np.float32)
            conc[good] = norm
            vdir = np.zeros((len(rows), 3), np.float32)
            vdir[good] = mean / norm[:, None]
    keep = conc >= conc_min
    vrows = rows[keep]
    types = sorted(set(cell[(ann.super_class == "visual_projection").to_numpy() & ok]))
    t_index = {t: i for i, t in enumerate(types)}
    vcell = cell[(ann.super_class == "visual_projection").to_numpy() & ok][keep]
    vside = side[(ann.super_class == "visual_projection").to_numpy() & ok][keep]
    cd = np.concatenate(col_dir)
    ce = np.concatenate(col_eye)
    # nearest column to each receptive field centre, in each eye; a neuron belongs to the eye whose columns
    # lie nearest its centre (a one-sided receptive field)
    dist = np.empty((len(vrows), 2), np.float32)
    best = np.empty((len(vrows), 2), np.int64)
    for code in (0, 1):
        members = np.flatnonzero(ce == code)
        d, i = cKDTree(cd[members]).query(vdir[keep])
        dist[:, code], best[:, code] = d, members[i]
    eye_of = np.argmin(dist, axis=1)
    rf_col = best[np.arange(len(vrows)), eye_of].astype(np.int32)
    return dict(col_dir=cd, col_eye=ce, rf_row=vrows.astype(np.int32), rf_id=net.ids[vrows],
                rf_dir=vdir[keep], rf_conc=conc[keep], rf_type=np.array([t_index[t] for t in vcell], np.int16),
                rf_col=rf_col, rf_side=np.array([{"left": 0, "right": 1}.get(s, 2) for s in vside], np.uint8),
                types=np.array(types))
