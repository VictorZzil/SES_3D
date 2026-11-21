"""SESAD inference from aligned descriptors and registered query points."""

from argparse import ArgumentParser

import numpy as np


SCALES = (1, 4, 16, 64)


def _unit_vectors(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values).all(axis=-1)
    clean = np.where(finite[..., None], values, 0.0)
    lengths = np.linalg.norm(clean, axis=-1)
    valid = finite & np.isfinite(lengths) & (lengths > 0)
    return np.divide(clean, lengths[..., None], out=np.zeros_like(clean),
                     where=valid[..., None]), valid


def project_map(points, centers, scores, sigma=1.0):
    """Apply normalized Gaussian interpolation with the paper's zero-weight fallback."""
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be positive and finite")
    point_map = np.zeros(len(points), dtype=np.float64)
    for start in range(0, len(points), 1024):
        batch = points[start:start + 1024]
        difference = batch[:, None, :] - centers[None, :, :]
        squared = np.einsum("bcd,bcd->bc", difference, difference)
        weights = np.exp(-squared / (2.0 * sigma ** 2))
        normalizer = weights.sum(axis=1)
        point_map[start:start + len(batch)] = np.divide(
            weights @ scores, normalizer, out=np.zeros(len(batch)), where=normalizer > 0
        )
    return point_map / (np.sqrt(2.0 * np.pi) * sigma)


def matched_excess(unit_bank, query_distances):
    """Evaluate matched leave-one-out evidence at full-support slots."""
    refs = unit_bank.shape[1]
    if refs < 2:
        raise ValueError("matched evidence requires at least two references")
    reference_distances = 1.0 - np.einsum("sid,sjd->sij", unit_bank, unit_bank)
    excess = []
    for omitted in range(refs):
        keep = np.arange(refs) != omitted
        normal_radius = reference_distances[:, omitted, keep].min(axis=1)
        query_distance = query_distances[:, keep].min(axis=1)
        excess.append(query_distance - normal_radius)
    return np.maximum(np.min(excess, axis=0), 0.0)


def spatial_score(centers, evidence):
    """Maximize normalized regional evidence over the canonical template."""
    if len(centers) == 0:
        return np.nan
    scales = np.minimum(SCALES, len(centers))
    best = 0.0
    for start in range(0, len(centers), 128):
        delta = centers[start:start + 128, None, :] - centers[None, :, :]
        squared = np.einsum("ijd,ijd->ij", delta, delta)
        row = np.arange(len(squared))
        squared[row, start + row] = -1.0  # The center is always a member.
        neighbors = np.argsort(squared, axis=1, kind="stable")[:, :scales[-1]]
        totals = evidence[neighbors].cumsum(axis=1)
        best = max(best, (totals[:, scales - 1] / np.sqrt(scales)).max())
    return best


def score_scan(points, query, bank, point_indices, canonical_centers,
               reference_valid=None, point_slot_mask=None, sigma=1.0):
    """Score a scan using (slots, dims) query and (slots, refs, dims) bank arrays.

    point_indices maps each template slot to its registered query point. Negative
    or out-of-range indices mark invalid correspondences. reference_valid can mask
    template entries with missing correspondences even if their descriptors are
    finite. point_slot_mask can preserve an upstream projection mask.
    """
    points = np.asarray(points, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    bank = np.asarray(bank, dtype=np.float64)
    point_indices = np.asarray(point_indices)
    canonical_centers = np.asarray(canonical_centers, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
            or not np.isfinite(points).all()):
        raise ValueError("points must be a nonempty finite (N, 3) array")
    if (query.ndim != 2 or bank.ndim != 3 or bank.shape[1] < 1
            or query.shape != (bank.shape[0], bank.shape[2])
            or query.shape[1] == 0):
        raise ValueError("query and bank must share slot and feature dimensions")
    slots = len(query)
    if point_indices.shape != (slots,) or point_indices.dtype.kind not in "iu":
        raise ValueError("point_indices must contain one integer index per slot")
    if canonical_centers.shape != (slots, 3):
        raise ValueError("canonical_centers must contain one 3D location per slot")

    unit_query, query_valid = _unit_vectors(query)
    unit_bank, bank_valid = _unit_vectors(bank)
    if reference_valid is not None:
        valid_mask = np.asarray(reference_valid, dtype=bool)
        if valid_mask.shape != bank_valid.shape:
            raise ValueError("reference_valid must have shape (slots, references)")
        bank_valid &= valid_mask
    eligible = query_valid & (point_indices >= 0) & (point_indices < len(points))
    distances = 1.0 - np.einsum("sd,std->st", unit_query, unit_bank)
    local_valid = eligible & bank_valid.any(axis=1)
    local = np.full(slots, np.nan)
    local[local_valid] = np.where(bank_valid, distances, np.inf)[local_valid].min(axis=1)
    projection_mask = local_valid.copy()
    if point_slot_mask is not None:
        mask = np.asarray(point_slot_mask, dtype=bool)
        if mask.shape != (slots,):
            raise ValueError("point_slot_mask must have shape (slots,)")
        projection_mask &= mask
    point_map = project_map(points, points[point_indices[projection_mask]],
                            local[projection_mask], sigma)

    object_mask = eligible & bank_valid.all(axis=1) & (bank.shape[1] >= 2)
    if not np.isfinite(canonical_centers[object_mask]).all():
        raise ValueError("canonical centers must be finite at object-scoring slots")
    evidence = np.full(slots, np.nan)
    if object_mask.any():
        evidence[object_mask] = matched_excess(unit_bank[object_mask], distances[object_mask])
    object_score = spatial_score(canonical_centers[object_mask], evidence[object_mask])
    return {
        "slot_scores": local,
        "point_scores": point_map,
        "point_max": point_map.max(),
        "slot_evidence": evidence,
        "spatial_score": object_score,
        "spatial_score_available": bool(object_mask.any()),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", help="registered points, aligned descriptors and slot metadata")
    parser.add_argument("output_npz", help="output score arrays")
    parser.add_argument("--sigma", type=float, default=1.0)
    args = parser.parse_args()
    with np.load(args.input_npz, allow_pickle=False) as entry:
        names = ("points", "query", "bank", "point_indices", "canonical_centers")
        outputs = score_scan(
            *(entry[key] for key in names),
            reference_valid=entry.get("reference_valid"),
            point_slot_mask=entry.get("point_slot_mask"),
            sigma=args.sigma,
        )
    np.savez_compressed(args.output_npz, **outputs)


if __name__ == "__main__":
    main()
