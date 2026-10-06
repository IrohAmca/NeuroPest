"""Viewing directions of the photoreceptors: local surface normals, body axes, sanity checks.

Run: uv run python tools/explore_eye2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from neuropest.paths import RAW_DIR

COLS = ["root_id", "super_class", "cell_class", "cell_type", "side", "pos_x", "pos_y", "pos_z"]
SC = np.array([4.0, 4.0, 40.0]) / 1000.0         # FlyWire voxels (4,4,40 nm) -> micrometres


def local_normals(p, k=25):
    """Outward-agnostic surface normals of points p: smallest principal axis of the k nearest neighbours."""
    tree = cKDTree(p)
    _, nn = tree.query(p, k=k)
    q = p[nn] - p[nn].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", q, q)
    w, v = np.linalg.eigh(cov)
    return v[:, :, 0], w                              # normal = eigenvector of the smallest eigenvalue


def main():
    ann = pd.read_csv(RAW_DIR / "Supplemental_file1_neuron_annotations.tsv", sep="\t", usecols=COLS, low_memory=False)
    ann = ann.dropna(subset=["pos_x"])
    P = ann[["pos_x", "pos_y", "pos_z"]].to_numpy(float) * SC

    def centroid(mask):
        return P[mask.to_numpy()].mean(0)

    pr = ann.cell_type.isin(["R1-6", "R7", "R8"]) & (ann.cell_class == "visual")
    left_eye = centroid(pr & (ann.side == "left"))
    right_eye = centroid(pr & (ann.side == "right"))
    ventral = centroid(ann.cell_class == "gustatory")
    dorsal = centroid(ann.cell_class.isin(["CX", "Kenyon_Cell"]))
    anterior = centroid(ann.cell_class == "olfactory")
    posterior = centroid(ann.super_class == "optic") * 0 + centroid(ann.cell_class == "Kenyon_Cell")
    R = right_eye - left_eye
    R /= np.linalg.norm(R)
    D = dorsal - ventral
    D -= R * (D @ R)
    D /= np.linalg.norm(D)
    A = anterior - posterior
    A -= R * (A @ R) + D * (A @ D)
    A /= np.linalg.norm(A)
    print("axes in data coordinates (right, dorsal, anterior):\n", np.round(np.stack([R, D, A]), 2))
    print("handedness A x R . D  (a true fly gives -1):", round(float(np.cross(A, R) @ D), 2))
    centre = (left_eye + right_eye) / 2
    for side in ("left", "right"):
        m = (pr & (ann.side == side)).to_numpy()
        p = P[m]
        n, w = local_normals(p)
        sign = np.sign((n * (p - p.mean(0))).sum(1))             # orient normals away from the eye's own centroid? no:
        # orient away from the brain centre (the point halfway between the two eye centroids shifted to the OL)
        sign = np.sign((n * (p - centre)).sum(1))
        n = n * sign[:, None]
        body = np.stack([n @ A, n @ R, n @ D], 1)                  # anterior, right, dorsal components
        az = np.degrees(np.arctan2(body[:, 1], body[:, 0]))      # 0 = anterior, +90 = right
        el = np.degrees(np.arcsin(np.clip(body[:, 2], -1, 1)))
        print(f"\n{side} eye ({m.sum()} photoreceptors): azimuth mean {az.mean():6.1f} sd {az.std():5.1f} "
              f"range [{np.percentile(az, 2):6.1f}, {np.percentile(az, 98):6.1f}]; elevation mean {el.mean():6.1f} "
              f"range [{np.percentile(el, 2):6.1f}, {np.percentile(el, 98):6.1f}]; normal flatness {np.median(w[:, 0] / w[:, 1]):.3f}")
        print("   centroid in body frame (anterior, right, dorsal, um):",
              np.round([(p.mean(0) - centre) @ A, (p.mean(0) - centre) @ R, (p.mean(0) - centre) @ D], 0))


if __name__ == "__main__":
    main()
