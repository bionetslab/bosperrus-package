"""Private readers for the raw/filtered outputs of 10x Space Ranger (Visium HD)
and BGI SAW (STOmics/Stereo-seq), used by `anndata_api.load_filtered` and
`anndata_api.quantify_diffusion_from_raw`.

Everything here only extracts per-bin *total* counts (summed over genes) plus
each bin's position -- never a full gene x bin matrix, which at 2um (Visium
HD) or 0.5um (Stereo-seq) would not fit in memory.

Heavy dependencies (h5py, pyarrow via pandas, tifffile) are imported lazily,
inside the functions that need them, so `bosperrus` stays importable without
the optional `raw` extra (`pip install bosperrus[raw]`).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

VISIUM_HD_RESOLUTIONS_UM = (2.0, 8.0, 16.0)
STEREOSEQ_DNB_PITCH_UM = 0.5
_TECHNOLOGIES = ("visium_hd", "stereo-seq")

# rows of a SAW gef `expression` dataset read per chunk (~180 MB per chunk)
_GEF_CHUNK_ROWS = 20_000_000


def _require(module, extra="raw"):
    try:
        return __import__(module)
    except ImportError:
        raise ImportError(
            f"{module} is required for this function. "
            f"Install it with: pip install {module} or pip install bosperrus[{extra}]"
        )


def validate_technology_and_resolution(technology, resolution):
    """Returns (technology, resolution_um, dnbs_per_bin) -- dnbs_per_bin is the
    Stereo-seq pooling factor (bin side length in DNBs), None for Visium HD.

    Visium HD: only the square bin sizes Space Ranger itself ships (2/8/16um).
    Stereo-seq: any integer multiple of the 0.5um DNB pitch, pooled from bin1
    by floor division exactly like SAW's own binN (so 10um == SAW bin20)."""
    if not isinstance(technology, str) or technology.lower() not in _TECHNOLOGIES:
        raise ValueError(f"technology must be one of {_TECHNOLOGIES}, got {technology!r}.")
    technology = technology.lower()
    resolution = float(resolution)
    if technology == "visium_hd":
        if resolution not in VISIUM_HD_RESOLUTIONS_UM:
            raise ValueError(
                f"Visium HD resolution must be one of {VISIUM_HD_RESOLUTIONS_UM} um "
                f"(the bin sizes Space Ranger ships), got {resolution}."
            )
        return technology, resolution, None
    dnbs_per_bin = resolution / STEREOSEQ_DNB_PITCH_UM
    if dnbs_per_bin < 1 or not np.isclose(dnbs_per_bin, round(dnbs_per_bin)):
        raise ValueError(
            f"Stereo-seq resolution must be a positive integer multiple of the "
            f"{STEREOSEQ_DNB_PITCH_UM}um DNB pitch, got {resolution}."
        )
    return technology, resolution, int(round(dnbs_per_bin))


# ---------------------------------------------------------------------------
# Visium HD (Space Ranger binned_outputs/)
# ---------------------------------------------------------------------------

def visium_bin_dir(path, resolution):
    """`path` is Space Ranger's `binned_outputs/` directory (its parent `outs/`
    is accepted too)."""
    path = Path(path)
    if (path / "binned_outputs").is_dir():
        path = path / "binned_outputs"
    bin_dir = path / f"square_{int(resolution):03d}um"
    if not bin_dir.is_dir():
        raise FileNotFoundError(
            f"{bin_dir} not found -- expected `path` to be a Space Ranger binned_outputs/ "
            f"directory containing square_{int(resolution):03d}um/."
        )
    return bin_dir


def read_10x_h5_total_counts(h5_path):
    """Per-barcode total counts from a 10x `*_feature_bc_matrix.h5` (CSC,
    features x barcodes), without building the matrix: column sums via a
    chunked cumulative sum over `data` evaluated at `indptr`.

    Returns (barcodes: np.ndarray of str, n_counts: np.ndarray of float64)."""
    h5py = _require("h5py")
    with h5py.File(h5_path, "r") as f:
        m = f["matrix"]
        barcodes = m["barcodes"][:].astype(str)
        indptr = m["indptr"][:].astype(np.int64)
        data = m["data"]
        nnz = data.shape[0]
        # cum[k] = sum(data[:k]); column j's total = cum[indptr[j+1]] - cum[indptr[j]]
        cum_at = np.zeros(len(indptr), dtype=np.float64)
        carry = 0.0
        for start in range(0, nnz, _GEF_CHUNK_ROWS):
            stop = min(start + _GEF_CHUNK_ROWS, nnz)
            chunk_cum = carry + np.cumsum(data[start:stop], dtype=np.float64)
            sel = (indptr > start) & (indptr <= stop)
            cum_at[sel] = chunk_cum[indptr[sel] - start - 1]
            carry = chunk_cum[-1]
    n_counts = np.diff(cum_at)
    if len(n_counts) != len(barcodes):
        raise ValueError(f"{h5_path}: indptr/barcodes length mismatch.")
    return barcodes, n_counts


def read_visium_positions(bin_dir, barcodes):
    """`tissue_positions.parquet` rows aligned to `barcodes` (raises if any
    barcode is missing)."""
    positions = pd.read_parquet(Path(bin_dir) / "spatial" / "tissue_positions.parquet")
    positions = positions.set_index("barcode")
    idx = positions.index.get_indexer(barcodes)
    if (idx < 0).any():
        raise ValueError(f"{(idx < 0).sum()} barcodes missing from tissue_positions.parquet in {bin_dir}.")
    return positions.iloc[idx]


def read_visium_scalefactors(bin_dir):
    with open(Path(bin_dir) / "spatial" / "scalefactors_json.json") as f:
        return json.load(f)


def read_visium_hires_image(bin_dir):
    """Space Ranger's own downsampled hires tissue image, as float RGB in [0, 1]."""
    import matplotlib.image as mpimg
    image = mpimg.imread(Path(bin_dir) / "spatial" / "tissue_hires_image.png")
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    return image


# ---------------------------------------------------------------------------
# Stereo-seq (SAW run folder)
# ---------------------------------------------------------------------------

def saw_outs_and_sn(path):
    """`path` is a SAW run folder (containing `outs/`) or its `outs/` itself.
    The chip serial number (SN) is read off the single `*.raw.gef`."""
    path = Path(path)
    outs = path / "outs" if (path / "outs").is_dir() else path
    raw_gefs = sorted((outs / "feature_expression").glob("*.raw.gef"))
    if len(raw_gefs) != 1:
        raise FileNotFoundError(
            f"Expected exactly one *.raw.gef in {outs / 'feature_expression'}, found "
            f"{[p.name for p in raw_gefs]} -- is `path` a SAW run folder?"
        )
    sn = raw_gefs[0].name[: -len(".raw.gef")]
    return outs, sn


def saw_chip_extent(outs, sn):
    """(min_y, min_x, len_y, len_x) of the chip's DNB grid in bin1 units, from
    `{SN}.tissue.gef`'s `wholeExp/bin1` attributes (the raw gef carries no
    wholeExp group). Checked against the raw data and `*_ssDNA_regist.tif` on
    real SAW output: identical extents."""
    h5py = _require("h5py")
    with h5py.File(Path(outs) / "feature_expression" / f"{sn}.tissue.gef", "r") as f:
        attrs = f["wholeExp"]["bin1"].attrs
        return (int(attrs["minY"][0]), int(attrs["minX"][0]), int(attrs["lenY"][0]), int(attrs["lenX"][0]))


def pool_gef_counts(gef_path, dnbs_per_bin, extent):
    """Total counts per pooled bin from a gef's `geneExp/bin1/expression`
    (x, y, count) records, pooled by floor division (bin = (y - min_y) // N,
    (x - min_x) // N), exactly like SAW's own binN. Partial bins at the far
    chip edge (chip length not a multiple of N) are dropped.

    Returns a dense (n_bin_rows, n_bin_cols) uint32 array -- zeros included,
    i.e. every grid position of the chip, not just positions with counts."""
    h5py = _require("h5py")
    min_y, min_x, len_y, len_x = extent
    n_rows, n_cols = len_y // dnbs_per_bin, len_x // dnbs_per_bin
    dense = np.zeros(n_rows * n_cols, dtype=np.uint32)
    with h5py.File(gef_path, "r") as f:
        expr = f["geneExp"]["bin1"]["expression"]
        for start in range(0, expr.shape[0], _GEF_CHUNK_ROWS):
            rec = expr[start:start + _GEF_CHUNK_ROWS]
            x = rec["x"].astype(np.int64) - min_x
            y = rec["y"].astype(np.int64) - min_y
            if (x < 0).any() or (y < 0).any() or (x >= len_x).any() or (y >= len_y).any():
                raise ValueError(f"{gef_path}: DNB coordinates outside the chip extent {extent}.")
            r, c = y // dnbs_per_bin, x // dnbs_per_bin
            keep = (r < n_rows) & (c < n_cols)
            flat = r[keep] * n_cols + c[keep]
            uniq, inv = np.unique(flat, return_inverse=True)
            dense[uniq] += np.bincount(inv, weights=rec["count"][keep]).astype(np.uint32)
    return dense.reshape(n_rows, n_cols)


def read_ssdna_image(outs, sn):
    tifffile = _require("tifffile")
    return tifffile.imread(Path(outs) / "image" / f"{sn}_ssDNA_regist.tif")
