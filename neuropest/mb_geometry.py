"""FlyWire v783 3D Mushroom Body Geometry & Anatomical Model.

Extracts, normalizes, and packages the 3D spatial coordinates of Kenyon cells,
MBONs, Dopaminergic neurons (PAM/PPL1), and the Drosophila brain wireframe.
Provides isotropic coordinate normalization, pedunculus trajectories, and
lobe compartment spatial centroids.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .paths import CIRCUITS, MUSHROOM, MUSHROOM_3D, RAW_DIR

# FAFB / FlyWire coordinate centering & normalization constants
# Original voxels: 4 nm x 4 nm x 40 nm. To make isotropic, z is multiplied by 10.
MIDLINE_X = 133200.0
CENTER_Y = 38000.0
CENTER_Z = 28000.0
NORM_SCALE = 38000.0

LOBE_GAMMA = 0
LOBE_ALPHABETA = 1
LOBE_ALPHAP_BETAP = 2

DAN_PAM = 0
DAN_PPL1 = 1
DAN_OTHER = 2


@dataclass
class Mushroom3DData:
    kc_soma: np.ndarray       # (N_kc, 3) float32: Calyx soma locations
    kc_ped: np.ndarray        # (N_kc, 3) float32: Pedunculus waypoint locations
    kc_target: np.ndarray     # (N_kc, 3) float32: Lobe terminal / compartment locations
    kc_lobe: np.ndarray       # (N_kc,) int8: 0=gamma, 1=ab, 2=apbp
    kc_side: np.ndarray       # (N_kc,) int8: -1=left, +1=right
    kc_type: np.ndarray       # (N_kc,) str
    kc2000_idx: np.ndarray    # (2000,) int32: representative subset for random model
    mbon_pos: np.ndarray      # (N_mbon, 3) float32: MBON terminal positions
    mbon_type: np.ndarray     # (N_mbon,) str: MBON01..MBON35
    mbon_side: np.ndarray     # (N_mbon,) int8: -1=left, +1=right
    mbon_sign: np.ndarray     # (N_mbon,) int8: +1 approach, -1 avoidance, 0 neutral
    mbon_nt: np.ndarray       # (N_mbon,) str: ACh, GABA, Glu
    dan_pos: np.ndarray       # (N_dan, 3) float32: Dopamine neuron locations
    dan_type: np.ndarray      # (N_dan,) str: PAM*, PPL1*, etc.
    dan_side: np.ndarray      # (N_dan,) int8: -1=left, +1=right
    dan_cluster: np.ndarray   # (N_dan,) int8: 0=PAM, 1=PPL1, 2=other
    brain_wireframe: np.ndarray # (N_lines, 2, 3) float32: Drosophila brain wireframe segments
    lobe_centers: dict[str, tuple[float, float, float]]

    @property
    def n_kc(self) -> int:
        return len(self.kc_soma)

    @property
    def n_mbon(self) -> int:
        return len(self.mbon_pos)

    @property
    def n_dan(self) -> int:
        return len(self.dan_pos)


def norm3d(pts: np.ndarray) -> np.ndarray:
    """Normalize isotropic FAFB coordinates into [-1.2, +1.2] range centered at brain midline."""
    p = np.empty_like(pts, dtype=np.float32)
    p[:, 0] = (pts[:, 0] - MIDLINE_X) / NORM_SCALE
    p[:, 1] = (pts[:, 1] - CENTER_Y) / NORM_SCALE
    p[:, 2] = (pts[:, 2] - CENTER_Z) / NORM_SCALE
    return p


def build_mushroom_3d(raw_dir: Path = RAW_DIR,
                      mushroom_path: Path = MUSHROOM,
                      out_path: Path = MUSHROOM_3D) -> Mushroom3DData:
    """Extract real FlyWire coordinates and build compressed 3D geometry npz."""
    import pandas as pd
    from scipy.spatial import ConvexHull

    m = np.load(mushroom_path)
    ann_path = raw_dir / "Supplemental_file1_neuron_annotations.tsv"
    want = ["root_id", "super_class", "cell_class", "cell_type", "side", "top_nt",
            "pos_x", "pos_y", "pos_z", "soma_x", "soma_y", "soma_z"]
    head = pd.read_csv(ann_path, sep="\t", nrows=0).columns
    usecols = [c for c in want if c in head]
    df = pd.read_csv(ann_path, sep="\t", usecols=usecols, low_memory=False)
    df = df.drop_duplicates("root_id").set_index("root_id")

    # Scale z by 10.0 for isotropic 4nm voxels
    for col in ["pos_z", "soma_z"]:
        if col in df.columns:
            df[col] = df[col].astype(np.float32) * 10.0

    kcs = df.reindex(m["kc_id"]).copy()
    mbons = df.reindex(m["mbon_id"]).copy()
    dans = df.reindex(m["dan_id"]).copy()

    # Fallbacks for any missing somas
    kcs["soma_x"] = kcs["soma_x"].fillna(kcs["pos_x"])
    kcs["soma_y"] = kcs["soma_y"].fillna(kcs["pos_y"])
    kcs["soma_z"] = kcs["soma_z"].fillna(kcs["pos_z"])

    kc_soma_norm = norm3d(kcs[["soma_x", "soma_y", "soma_z"]].to_numpy())
    kc_pos_norm = norm3d(kcs[["pos_x", "pos_y", "pos_z"]].to_numpy())
    mbon_pos_norm = norm3d(mbons[["pos_x", "pos_y", "pos_z"]].to_numpy())
    dan_pos_norm = norm3d(dans[["pos_x", "pos_y", "pos_z"]].to_numpy())

    # Calculate target MBON positions for KCs (arborization in specific lobe compartments)
    kc_to_mbon = m["kc_mbon_mbon"]
    kc_idx = m["kc_mbon_kc"]
    mbon_pos_raw = mbons[["pos_x", "pos_y", "pos_z"]].to_numpy()

    kc_target_raw = np.zeros((len(kcs), 3), np.float32)
    kc_target_cnt = np.zeros(len(kcs), np.int32)
    np.add.at(kc_target_raw, kc_idx, mbon_pos_raw[kc_to_mbon])
    np.add.at(kc_target_cnt, kc_idx, 1)

    has_target = kc_target_cnt > 0
    kc_target_raw[has_target] /= kc_target_cnt[has_target, None]
    kc_target_raw[~has_target] = kcs[["pos_x", "pos_y", "pos_z"]].to_numpy()[~has_target]
    kc_target_norm = norm3d(kc_target_raw)

    # Classify KC lobes: 0=gamma, 1=ab, 2=apbp
    kc_types = kcs["cell_type"].fillna("").to_numpy(str)
    kc_lobe = np.zeros(len(kcs), np.int8)
    for i, t in enumerate(kc_types):
        if "KCg" in t:
            kc_lobe[i] = LOBE_GAMMA
        elif "KCab" in t:
            kc_lobe[i] = LOBE_ALPHABETA
        elif "Capbp" in t or "Ca'b'" in t:
            kc_lobe[i] = LOBE_ALPHAP_BETAP

    kc_side = np.where(kcs["side"] == "right", 1, -1).astype(np.int8)

    # Pedunculus corridor waypoint: connects Calyx down to anterior heel
    heel_x = kc_side * 0.48
    heel_y = np.full(len(kcs), 0.10, dtype=np.float32)
    heel_z = np.full(len(kcs), -0.12, dtype=np.float32)
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.025, (len(kcs), 3)).astype(np.float32)
    kc_ped_norm = np.column_stack([heel_x, heel_y, heel_z]).astype(np.float32) + jitter

    # MBON valence sign
    mbon_types = mbons["cell_type"].fillna("").to_numpy(str)
    mbon_side = np.where(mbons["side"] == "right", 1, -1).astype(np.int8)
    mbon_nt = mbons["top_nt"].fillna("").to_numpy(str)
    from .mushroom import MBON_VALENCE_LITERATURE
    mbon_sign = np.zeros(len(mbons), np.int8)
    for i, t in enumerate(mbon_types):
        if t in MBON_VALENCE_LITERATURE:
            mbon_sign[i] = MBON_VALENCE_LITERATURE[t]
        elif mbon_nt[i] == "acetylcholine":
            mbon_sign[i] = 1
        elif mbon_nt[i] in ("gaba", "glutamate"):
            mbon_sign[i] = -1

    # DAN clusters
    dan_types = dans["cell_type"].fillna("").to_numpy(str)
    dan_side = np.where(dans["side"] == "right", 1, -1).astype(np.int8)
    dan_cluster = np.full(len(dans), DAN_OTHER, np.int8)
    for i, t in enumerate(dan_types):
        if t.startswith("PAM"):
            dan_cluster[i] = DAN_PAM
        elif t.startswith("PPL1"):
            dan_cluster[i] = DAN_PPL1

    # Representative 2,000 KC subset for random rate model
    # Maintain lobe balance: ~960 gamma, ~680 ab, ~360 apbp
    idx_gamma = np.flatnonzero(kc_lobe == LOBE_GAMMA)
    idx_ab = np.flatnonzero(kc_lobe == LOBE_ALPHABETA)
    idx_apbp = np.flatnonzero(kc_lobe == LOBE_ALPHAP_BETAP)
    n_g = min(960, len(idx_gamma))
    n_ab = min(680, len(idx_ab))
    n_ap = min(360, len(idx_apbp))
    kc2000 = np.concatenate([
        rng.choice(idx_gamma, n_g, replace=False),
        rng.choice(idx_ab, n_ab, replace=False),
        rng.choice(idx_apbp, n_ap, replace=False),
    ])
    if len(kc2000) < 2000:
        rem = np.setdiff1d(np.arange(len(kcs)), kc2000)
        kc2000 = np.concatenate([kc2000, rng.choice(rem, 2000 - len(kc2000), replace=False)])
    kc2000_idx = kc2000.astype(np.int32)

    # Compute brain wireframe contours (horizontal convex hull rings of central brain + optic lobes)
    all_pts_raw = df[["pos_x", "pos_y", "pos_z"]].dropna().to_numpy()
    all_pts_norm = norm3d(all_pts_raw)
    levels = [-0.25, -0.05, 0.15, 0.40, 0.70]
    rings = []
    lines = []
    for y_lvl in levels:
        mask = (all_pts_norm[:, 1] >= y_lvl - 0.08) & (all_pts_norm[:, 1] <= y_lvl + 0.08)
        sub = all_pts_norm[mask]
        if len(sub) > 20:
            hull = ConvexHull(sub[:, [0, 2]])
            ring_verts = sub[hull.vertices]
            # Connect loop
            for vi in range(len(ring_verts)):
                p1 = ring_verts[vi]
                p2 = ring_verts[(vi + 1) % len(ring_verts)]
                lines.append((p1, p2))
            rings.append(ring_verts)

    # Add vertical connector ribs between adjacent rings
    for ri in range(len(rings) - 1):
        r1, r2 = rings[ri], rings[ri + 1]
        step = max(1, len(r1) // 12)
        for i in range(0, len(r1), step):
            p1 = r1[i]
            dists = np.sum((r2[:, [0, 2]] - p1[[0, 2]]) ** 2, axis=1)
            p2 = r2[np.argmin(dists)]
            lines.append((p1, p2))

    brain_wireframe = np.array(lines, dtype=np.float32)

    # Compute key lobe spatial centroids
    lobe_centers = {
        "calyx_left": tuple(np.mean(kc_soma_norm[kc_side == -1], axis=0)),
        "calyx_right": tuple(np.mean(kc_soma_norm[kc_side == 1], axis=0)),
        "gamma_left": tuple(np.mean(kc_target_norm[(kc_side == -1) & (kc_lobe == LOBE_GAMMA)], axis=0)),
        "gamma_right": tuple(np.mean(kc_target_norm[(kc_side == 1) & (kc_lobe == LOBE_GAMMA)], axis=0)),
        "alpha_left": tuple(np.mean(kc_target_norm[(kc_side == -1) & (kc_lobe == LOBE_ALPHABETA)], axis=0)),
        "alpha_right": tuple(np.mean(kc_target_norm[(kc_side == 1) & (kc_lobe == LOBE_ALPHABETA)], axis=0)),
        "pam_reward": tuple(np.mean(dan_pos_norm[dan_cluster == DAN_PAM], axis=0)),
        "ppl1_punish": tuple(np.mean(dan_pos_norm[dan_cluster == DAN_PPL1], axis=0)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        kc_soma=kc_soma_norm,
        kc_ped=kc_ped_norm,
        kc_target=kc_target_norm,
        kc_lobe=kc_lobe,
        kc_side=kc_side,
        kc_type=kc_types,
        kc2000_idx=kc2000_idx,
        mbon_pos=mbon_pos_norm,
        mbon_type=mbon_types,
        mbon_side=mbon_side,
        mbon_sign=mbon_sign,
        mbon_nt=mbon_nt,
        dan_pos=dan_pos_norm,
        dan_type=dan_types,
        dan_side=dan_side,
        dan_cluster=dan_cluster,
        brain_wireframe=brain_wireframe,
        lobe_centers_keys=np.array(list(lobe_centers.keys())),
        lobe_centers_vals=np.array(list(lobe_centers.values()), dtype=np.float32),
    )

    return Mushroom3DData(
        kc_soma=kc_soma_norm,
        kc_ped=kc_ped_norm,
        kc_target=kc_target_norm,
        kc_lobe=kc_lobe,
        kc_side=kc_side,
        kc_type=kc_types,
        kc2000_idx=kc2000_idx,
        mbon_pos=mbon_pos_norm,
        mbon_type=mbon_types,
        mbon_side=mbon_side,
        mbon_sign=mbon_sign,
        mbon_nt=mbon_nt,
        dan_pos=dan_pos_norm,
        dan_type=dan_types,
        dan_side=dan_side,
        dan_cluster=dan_cluster,
        brain_wireframe=brain_wireframe,
        lobe_centers=lobe_centers,
    )


def build_fallback_model() -> Mushroom3DData:
    """Procedural anatomical fallback if raw connectome annotations are absent."""
    n_kc = 2000
    rng = np.random.default_rng(7)
    sides = np.where(rng.uniform(0, 1, n_kc) > 0.5, 1, -1).astype(np.int8)

    # Calyx somas (posterior cup)
    calyx_r = rng.uniform(0.08, 0.28, n_kc).astype(np.float32)
    calyx_th = rng.uniform(0, 2 * math.pi, n_kc).astype(np.float32)
    soma_x = sides * 0.62 + calyx_r * np.cos(calyx_th)
    soma_y = -0.15 + calyx_r * np.sin(calyx_th) * 0.7
    soma_z = 0.50 + rng.uniform(-0.1, 0.15, n_kc).astype(np.float32)
    kc_soma = np.column_stack([soma_x, soma_y, soma_z]).astype(np.float32)

    # Pedunculus waypoint
    ped_x = sides * 0.48 + rng.normal(0, 0.02, n_kc).astype(np.float32)
    ped_y = np.full(n_kc, 0.10, dtype=np.float32) + rng.normal(0, 0.02, n_kc).astype(np.float32)
    ped_z = np.full(n_kc, -0.12, dtype=np.float32) + rng.normal(0, 0.02, n_kc).astype(np.float32)
    kc_ped = np.column_stack([ped_x, ped_y, ped_z]).astype(np.float32)

    # Lobes
    lobes = rng.choice([LOBE_GAMMA, LOBE_ALPHABETA, LOBE_ALPHAP_BETAP], n_kc, p=[0.48, 0.34, 0.18]).astype(np.int8)
    target_x = np.empty(n_kc, dtype=np.float32)
    target_y = np.empty(n_kc, dtype=np.float32)
    target_z = np.empty(n_kc, dtype=np.float32)

    # Gamma: horizontal medial
    is_g = lobes == LOBE_GAMMA
    target_x[is_g] = sides[is_g] * rng.uniform(0.10, 0.35, is_g.sum()).astype(np.float32)
    target_y[is_g] = 0.18 + rng.normal(0, 0.03, is_g.sum()).astype(np.float32)
    target_z[is_g] = -0.35 + rng.normal(0, 0.04, is_g.sum()).astype(np.float32)

    # Alpha/Beta: vertical & medial
    is_ab = lobes == LOBE_ALPHABETA
    target_x[is_ab] = sides[is_ab] * rng.uniform(0.35, 0.55, is_ab.sum()).astype(np.float32)
    target_y[is_ab] = -0.18 + rng.normal(0, 0.04, is_ab.sum()).astype(np.float32)
    target_z[is_ab] = -0.05 + rng.normal(0, 0.05, is_ab.sum()).astype(np.float32)

    # Alpha'/Beta'
    is_ap = lobes == LOBE_ALPHAP_BETAP
    target_x[is_ap] = sides[is_ap] * rng.uniform(0.30, 0.50, is_ap.sum()).astype(np.float32)
    target_y[is_ap] = -0.12 + rng.normal(0, 0.03, is_ap.sum()).astype(np.float32)
    target_z[is_ap] = -0.08 + rng.normal(0, 0.04, is_ap.sum()).astype(np.float32)

    kc_target = np.column_stack([target_x, target_y, target_z]).astype(np.float32)

    # Synthetic MBONs (48 per hemisphere)
    n_mbon = 96
    m_sides = np.array([-1] * 48 + [1] * 48, dtype=np.int8)
    m_pos = np.column_stack([
        m_sides * rng.uniform(0.15, 0.50, n_mbon).astype(np.float32),
        rng.uniform(-0.15, 0.25, n_mbon).astype(np.float32),
        rng.uniform(-0.35, 0.05, n_mbon).astype(np.float32)
    ])
    m_types = np.array([f"MBON{i:02d}" for i in range(1, n_mbon + 1)])
    m_sign = rng.choice([-1, 1], n_mbon, p=[0.6, 0.4]).astype(np.int8)
    m_nt = rng.choice(["acetylcholine", "gaba", "glutamate"], n_mbon)

    # Synthetic DANs (PAM reward, PPL1 punishment)
    n_dan = 120
    d_sides = np.array([-1] * 60 + [1] * 60, dtype=np.int8)
    d_cluster = rng.choice([DAN_PAM, DAN_PPL1], n_dan, p=[0.75, 0.25]).astype(np.int8)
    d_pos = np.empty((n_dan, 3), np.float32)
    for i in range(n_dan):
        if d_cluster[i] == DAN_PAM: # Medial lobe
            d_pos[i] = [d_sides[i] * rng.uniform(0.12, 0.32), 0.16 + rng.normal(0, 0.03), -0.32 + rng.normal(0, 0.03)]
        else: # Vertical lobe / heel
            d_pos[i] = [d_sides[i] * rng.uniform(0.40, 0.55), -0.05 + rng.normal(0, 0.04), -0.02 + rng.normal(0, 0.03)]
    d_types = np.array([f"PAM{i}" if c == DAN_PAM else f"PPL1_{i}" for i, c in enumerate(d_cluster)])

    # Synthetic brain wireframe cage
    rings_levels = [-0.25, -0.05, 0.15, 0.40, 0.70]
    wf_lines = []
    for y_lvl in rings_levels:
        n_pts = 16
        ths = np.linspace(0, 2 * math.pi, n_pts, endpoint=False)
        pts = np.column_stack([1.1 * np.cos(ths), np.full(n_pts, y_lvl), 0.7 * np.sin(ths)]).astype(np.float32)
        for vi in range(n_pts):
            wf_lines.append((pts[vi], pts[(vi + 1) % n_pts]))
    brain_wireframe = np.array(wf_lines, dtype=np.float32)

    lobe_centers = {
        "calyx_left": (-0.62, -0.15, 0.50),
        "calyx_right": (0.62, -0.15, 0.50),
        "gamma_left": (-0.22, 0.18, -0.35),
        "gamma_right": (0.22, 0.18, -0.35),
        "alpha_left": (-0.45, -0.18, -0.05),
        "alpha_right": (0.45, -0.18, -0.05),
        "pam_reward": (0.22, 0.16, -0.32),
        "ppl1_punish": (0.48, -0.05, -0.02),
    }

    return Mushroom3DData(
        kc_soma=kc_soma,
        kc_ped=kc_ped,
        kc_target=kc_target,
        kc_lobe=lobes,
        kc_side=sides,
        kc_type=np.array(["KC" for _ in range(n_kc)]),
        kc2000_idx=np.arange(n_kc, dtype=np.int32),
        mbon_pos=m_pos,
        mbon_type=m_types,
        mbon_side=m_sides,
        mbon_sign=m_sign,
        mbon_nt=m_nt,
        dan_pos=d_pos,
        dan_type=d_types,
        dan_side=d_sides,
        dan_cluster=d_cluster,
        brain_wireframe=brain_wireframe,
        lobe_centers=lobe_centers,
    )


def load_mushroom_3d(path: Path = MUSHROOM_3D) -> Mushroom3DData:
    """Load cached 3D geometry npz or build it on demand."""
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                keys = z["lobe_centers_keys"].tolist()
                vals = [tuple(v) for v in z["lobe_centers_vals"]]
                lobe_centers = dict(zip(keys, vals))
                return Mushroom3DData(
                    kc_soma=z["kc_soma"],
                    kc_ped=z["kc_ped"],
                    kc_target=z["kc_target"],
                    kc_lobe=z["kc_lobe"],
                    kc_side=z["kc_side"],
                    kc_type=z["kc_type"],
                    kc2000_idx=z["kc2000_idx"],
                    mbon_pos=z["mbon_pos"],
                    mbon_type=z["mbon_type"],
                    mbon_side=z["mbon_side"],
                    mbon_sign=z["mbon_sign"],
                    mbon_nt=z["mbon_nt"],
                    dan_pos=z["dan_pos"],
                    dan_type=z["dan_type"],
                    dan_side=z["dan_side"],
                    dan_cluster=z["dan_cluster"],
                    brain_wireframe=z["brain_wireframe"],
                    lobe_centers=lobe_centers,
                )
        except (OSError, KeyError, ValueError):
            pass

    # Try building from raw FlyWire data if available
    if (RAW_DIR / "Supplemental_file1_neuron_annotations.tsv").exists() and MUSHROOM.exists():
        try:
            return build_mushroom_3d(RAW_DIR, MUSHROOM, path)
        except Exception:
            pass

    return build_fallback_model()
