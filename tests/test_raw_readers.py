"""load_filtered / quantify_diffusion_from_raw against tiny synthetic Space
Ranger (binned_outputs/) and SAW (run folder) outputs written to tmp_path,
with a known tissue disk and a known exponential decay of counts outside it."""
import json

import numpy as np
import pandas as pd
import pytest

anndata = pytest.importorskip("anndata", reason="anndata not installed")
h5py = pytest.importorskip("h5py", reason="h5py not installed")
pytest.importorskip("pyarrow", reason="pyarrow not installed")
tifffile = pytest.importorskip("tifffile", reason="tifffile not installed")
pytest.importorskip("skimage", reason="scikit-image not installed")

from bosperrus.anndata_api import (
    load_filtered, quantify_diffusion_from_raw, quantify_diffusion_visium_HD, quantify_diffusion_stereoseq,
    identify_analysis_buffer_from_filtered,
)

A_TRUE, B_TRUE = 150.0, 0.05  # counts per bin at the mask edge, decay rate (1/um)


# ---------------------------------------------------------------------------
# synthetic Visium HD
# ---------------------------------------------------------------------------

def _write_10x_h5(path, barcodes, n_counts, n_features=3):
    """CSC features x barcodes; each barcode's counts split over the features."""
    rng = np.random.default_rng(0)
    data, indices, indptr = [], [], [0]
    for total in n_counts.astype(int):
        split = rng.multinomial(total, np.ones(n_features) / n_features)
        nz = np.flatnonzero(split)
        data.extend(split[nz]); indices.extend(nz); indptr.append(indptr[-1] + len(nz))
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["data"] = np.asarray(data, dtype=np.int32)
        m["indices"] = np.asarray(indices, dtype=np.int64)
        m["indptr"] = np.asarray(indptr, dtype=np.int64)
        m["shape"] = np.array([n_features, len(barcodes)], dtype=np.int32)
        m["barcodes"] = np.asarray(barcodes, dtype="S")


def _make_visium(tmp_path, n_side=90, microns_per_pixel=0.5, hires_scalef=0.25, radius_um=150.0,
                 offset_fullres=-80.0):
    """8um bins on an n_side x n_side grid; hires image 320x320 px (2um/px),
    a dark tissue disk on white. The bin grid starts at `offset_fullres`
    (negative: outside the image, to exercise padding) and runs past the
    image's far edge too."""
    rng = np.random.default_rng(42)
    binned = tmp_path / "binned_outputs"
    bin_dir = binned / "square_008um"
    (bin_dir / "spatial").mkdir(parents=True)

    um_per_hires_px = microns_per_pixel / hires_scalef
    size = 320
    centre_um = size * um_per_hires_px / 2
    yy, xx = np.mgrid[:size, :size] * um_per_hires_px
    disk = np.hypot(yy - centre_um, xx - centre_um) <= radius_um
    image = np.ones((size, size, 3))
    image[disk] = 0.3
    import matplotlib.image as mpimg
    mpimg.imsave(bin_dir / "spatial" / "tissue_hires_image.png", image)

    pitch_fullres = 8.0 / microns_per_pixel
    rows, cols = np.meshgrid(np.arange(n_side), np.arange(n_side), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    pxl_row = offset_fullres + rows * pitch_fullres
    pxl_col = offset_fullres + cols * pitch_fullres
    y_um, x_um = pxl_row * microns_per_pixel, pxl_col * microns_per_pixel
    d_true = np.maximum(0, np.hypot(y_um - centre_um, x_um - centre_um) - radius_um)
    lam = np.where(d_true > 0, A_TRUE * np.exp(-B_TRUE * d_true), 400.0)
    n_counts = rng.poisson(lam).astype(float)

    barcodes = np.array([f"s_008um_{r:05d}_{c:05d}-1" for r, c in zip(rows, cols)])
    in_tissue = d_true == 0
    pd.DataFrame({
        "barcode": barcodes, "in_tissue": in_tissue.astype(int), "array_row": rows, "array_col": cols,
        "pxl_row_in_fullres": pxl_row, "pxl_col_in_fullres": pxl_col,
    }).to_parquet(bin_dir / "spatial" / "tissue_positions.parquet")
    with open(bin_dir / "spatial" / "scalefactors_json.json", "w") as f:
        json.dump({"microns_per_pixel": microns_per_pixel, "tissue_hires_scalef": hires_scalef, "bin_size_um": 8.0}, f)
    _write_10x_h5(bin_dir / "raw_feature_bc_matrix.h5", barcodes, n_counts)
    _write_10x_h5(bin_dir / "filtered_feature_bc_matrix.h5", barcodes[in_tissue], n_counts[in_tissue])
    return binned, dict(rows=rows, cols=cols, n_counts=n_counts, d_true=d_true, in_tissue=in_tissue)


def test_load_filtered_visium_reads_filtered_bins_and_positions(tmp_path):
    binned, truth = _make_visium(tmp_path)
    adata = load_filtered(binned, "visium_hd", 8)

    assert adata.n_obs == truth["in_tissue"].sum()
    assert adata.n_vars == 0
    np.testing.assert_array_equal(adata.obs["n_counts"].to_numpy(), truth["n_counts"][truth["in_tissue"]])
    np.testing.assert_array_equal(adata.obs["array_row"].to_numpy(), truth["rows"][truth["in_tissue"]])
    assert adata.uns["bosperrus"]["bin_size_um"] == 8.0
    assert adata.uns["bosperrus"]["technology"] == "visium_hd"


def test_load_filtered_visium_grid_indices_are_signed(tmp_path):
    """Space Ranger's parquet stores array_row/array_col as uint32, where
    `row - 1` silently wraps around; load_filtered must hand out signed ints."""
    binned, _ = _make_visium(tmp_path)
    positions = pd.read_parquet(binned / "square_008um" / "spatial" / "tissue_positions.parquet")
    positions[["array_row", "array_col"]] = positions[["array_row", "array_col"]].astype(np.uint32)
    positions.to_parquet(binned / "square_008um" / "spatial" / "tissue_positions.parquet")

    adata = load_filtered(binned, "visium_hd", 8)
    assert np.issubdtype(adata.obs["array_row"].dtype, np.signedinteger)
    assert (adata.obs["array_row"].to_numpy().min() - 1) < adata.obs["array_row"].to_numpy().min()


def test_load_filtered_accepts_outs_parent(tmp_path):
    binned, truth = _make_visium(tmp_path)
    adata = load_filtered(binned.parent, "visium_hd", 8)  # tmp_path contains binned_outputs/
    assert adata.n_obs == truth["in_tissue"].sum()


def test_quantify_diffusion_visium_recovers_decay_and_counts_zero_bins(tmp_path):
    binned, truth = _make_visium(tmp_path)
    result, data = quantify_diffusion_from_raw(binned, "visium_hd", 8, return_data=True)

    assert result["best_fit_type"] == "Exponential Decay Fit"
    assert result["beta"] == pytest.approx(B_TRUE, rel=0.15)
    assert result["alpha"] == pytest.approx(A_TRUE / 8.0 ** 2, rel=0.3)
    assert result["decay_length_um"] == pytest.approx(1 / result["beta"])
    # every raw bin counts, zero-count ones included
    assert result["n_bins_total"] == len(truth["n_counts"])
    assert (data["n_counts"][data["distance_um"] > 0] == 0).any()
    outside = data["distance_um"] > 0
    assert result["perc_counts_outside"] == pytest.approx(
        100 * truth["n_counts"][outside].sum() / truth["n_counts"].sum())
    json.dumps(result)  # JSON-serializable


def test_quantify_diffusion_visium_pads_bins_beyond_image(tmp_path):
    """Bins outside the hires image must get their true distance to the
    mask, not the distance of the nearest image-edge pixel (which is what
    distance_to_mask's clipping would give without padding)."""
    binned, truth = _make_visium(tmp_path)
    result, data = quantify_diffusion_visium_HD(binned, 8, return_data=True)

    beyond = (data["mask_row"] < 0) | (data["mask_col"] < 0) | \
             (data["mask_row"] > data["mask"].shape[0] - 1) | (data["mask_col"] > data["mask"].shape[1] - 1)
    assert beyond.any()
    np.testing.assert_allclose(data["distance_um"][beyond], truth["d_true"][beyond], atol=6.0)


# ---------------------------------------------------------------------------
# synthetic Stereo-seq (SAW run folder)
# ---------------------------------------------------------------------------

_GEF_DTYPE = np.dtype([("x", "<i4"), ("y", "<i4"), ("count", "u1")])


def _write_gef(path, x, y, count, whole_extent=None):
    with h5py.File(path, "w") as f:
        f.create_group("geneExp/bin1")["expression"] = np.rec.fromarrays([x, y, count], dtype=_GEF_DTYPE)
        if whole_extent is not None:
            len_y, len_x = whole_extent
            g = f.create_group("wholeExp").create_dataset("bin1", data=np.zeros(1))
            for k, v in {"lenX": len_x, "lenY": len_y, "minX": 0, "minY": 0}.items():
                g.attrs[k] = np.array([v], dtype=np.uint32)


def _make_saw(tmp_path, len_y=800, len_x=806, radius_dnb=150, sn="T00000A1"):
    """0.5um DNBs; a bright tissue disk (radius 75um) on a dark ssDNA image;
    counts at 4um resolution (8x8 DNBs), one gef record per bin placed on the
    bin's first DNB (split into <=255 chunks for the uint8 count field).
    len_x=806 leaves a 6-DNB partial bin column at the far edge."""
    rng = np.random.default_rng(42)
    outs = tmp_path / "run" / "outs"
    (outs / "feature_expression").mkdir(parents=True)
    (outs / "image").mkdir()

    cy, cx = len_y / 2, len_x / 2
    yy, xx = np.mgrid[:len_y, :len_x]
    image = np.where(np.hypot(yy - cy, xx - cx) <= radius_dnb, 200, 20).astype(np.uint8)
    tifffile.imwrite(outs / "image" / f"{sn}_ssDNA_regist.tif", image)

    n = 8
    by, bx = np.mgrid[:len_y // n, :len_x // n]
    by, bx = by.ravel(), bx.ravel()
    centre_y, centre_x = by * n + (n - 1) / 2, bx * n + (n - 1) / 2
    d_true = np.maximum(0, np.hypot(centre_y - cy, centre_x - cx) - radius_dnb) * 0.5
    lam = np.where(d_true > 0, A_TRUE * np.exp(-B_TRUE * d_true), 400.0)
    counts = rng.poisson(lam)

    xs, ys, cs = [], [], []
    for r, c, k in zip(by, bx, counts):
        while k > 0:
            chunk = min(k, 255)
            xs.append(c * n); ys.append(r * n); cs.append(chunk); k -= chunk
    xs.append(len_x - 2); ys.append(0); cs.append(99)  # in the partial edge column -> dropped
    xs, ys, cs = map(np.asarray, (xs, ys, cs))
    _write_gef(outs / "feature_expression" / f"{sn}.raw.gef", xs, ys, cs)

    in_tissue = np.isin(ys // n * (len_x // n) + xs // n, (by * (len_x // n) + bx)[d_true == 0]) & (xs < (len_x // n) * n)
    _write_gef(outs / "feature_expression" / f"{sn}.tissue.gef", xs[in_tissue], ys[in_tissue], cs[in_tissue],
               whole_extent=(len_y, len_x))
    return outs.parent, dict(by=by, bx=bx, counts=counts, d_true=d_true, n=n, sn=sn)


def test_load_filtered_stereoseq_pools_tissue_gef(tmp_path):
    run, truth = _make_saw(tmp_path)
    adata = load_filtered(run, "stereo-seq", 4)  # 4um = 8x8 DNBs

    tissue = truth["d_true"] == 0
    expected = pd.Series(truth["counts"][tissue], index=[f"{r}_{c}" for r, c in zip(truth["by"][tissue], truth["bx"][tissue])])
    expected = expected[expected > 0]  # gefs only list bins with counts
    assert adata.n_obs == len(expected)
    np.testing.assert_array_equal(adata.obs["n_counts"].reindex(expected.index).to_numpy(), expected.to_numpy())
    assert adata.uns["bosperrus"]["bin_size_um"] == 4.0


def test_load_filtered_stereoseq_coarser_pooling_conserves_counts(tmp_path):
    run, _ = _make_saw(tmp_path)
    fine = load_filtered(run, "stereo-seq", 4)
    coarse = load_filtered(run, "stereo-seq", 8)  # 16x16 DNBs; 806 // 16 = 50 full columns (800 DNBs), same as 806 // 8 = 100
    assert coarse.obs["n_counts"].sum() == fine.obs["n_counts"].sum()
    assert coarse.n_obs < fine.n_obs


def test_quantify_diffusion_stereoseq_recovers_decay_on_dense_grid(tmp_path):
    run, truth = _make_saw(tmp_path)
    result, data = quantify_diffusion_from_raw(
        run, "stereo-seq", 4, mask_kwargs={"target_size": 200, "close_radius": 2}, return_data=True,
    )

    assert result["sn"] == truth["sn"]
    assert result["best_fit_type"] == "Exponential Decay Fit"
    assert result["beta"] == pytest.approx(B_TRUE, rel=0.15)
    assert result["alpha"] == pytest.approx(A_TRUE / 4.0 ** 2, rel=0.3)
    # dense grid: every full bin of the chip, zeros included; partial edge column dropped
    assert result["n_bins_total"] == len(truth["counts"])
    assert result["n_counts_total"] == truth["counts"].sum()
    assert (data["n_counts"] == 0).any()
    assert result["mask_pixel_size_um"] == pytest.approx(0.5 * 4)  # factor = round(806 / 200) = 4
    np.testing.assert_allclose(data["distance_um"], truth["d_true"], atol=3.0)


def test_quantify_diffusion_stereoseq_rejects_image_chip_mismatch(tmp_path):
    run, truth = _make_saw(tmp_path)
    tifffile.imwrite(run / "outs" / "image" / f"{truth['sn']}_ssDNA_regist.tif", np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError, match="can't map bins"):
        quantify_diffusion_stereoseq(run, 4)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("technology,resolution", [("visium_hd", 10), ("visium_hd", 0.5), ("stereo-seq", 0.7), ("stereo-seq", 0)])
def test_invalid_resolution_raises(tmp_path, technology, resolution):
    with pytest.raises(ValueError, match="resolution"):
        quantify_diffusion_from_raw(tmp_path, technology, resolution)
    with pytest.raises(ValueError, match="resolution"):
        load_filtered(tmp_path, technology, resolution)


def test_invalid_technology_raises(tmp_path):
    with pytest.raises(ValueError, match="technology"):
        quantify_diffusion_from_raw(tmp_path, "xenium", 8)


def test_max_bins_guard(tmp_path):
    run, _ = _make_saw(tmp_path)
    with pytest.raises(ValueError, match="max_bins"):
        quantify_diffusion_stereoseq(run, 0.5, max_bins=1000)


def test_wrong_folder_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="raw.gef"):
        load_filtered(tmp_path, "stereo-seq", 8)
    with pytest.raises(FileNotFoundError, match="binned_outputs"):
        load_filtered(tmp_path, "visium_hd", 8)


def test_load_filtered_feeds_identify_analysis_buffer(tmp_path):
    binned, _ = _make_visium(tmp_path)
    adata = load_filtered(binned, "visium_hd", 8)
    identify_analysis_buffer_from_filtered(adata)
    assert adata.uns["analysis_buffer_fit"]["bin_size_um"] == 8.0
    assert "analysis_buffer" in adata.obs
