import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skimage", reason="scikit-image not installed")
anndata = pytest.importorskip("anndata", reason="anndata not installed")

from bosperrus.image_masks import get_hires_image, segment_tissue_from_rgb, segment_tissue_from_ssdna, get_tissue_mask


def _synthetic_tissue_image(size=200, blob_slice=slice(60, 140)):
    """White background, dark square 'tissue' blob in the center."""
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[blob_slice, blob_slice, :] = 30
    return image


def _adata_with_image(library_id="sample1", image=None, pixel_scale=0.2):
    if image is None:
        image = _synthetic_tissue_image()
    adata = anndata.AnnData(X=np.zeros((3, 2)), obs=pd.DataFrame(index=["a", "b", "c"]))
    adata.uns["spatial"] = {
        library_id: {
            "images": {"hires": image},
            "scalefactors": {"tissue_hires_scalef": pixel_scale},
        }
    }
    return adata


# ---------------------------------------------------------------------------
# segment_tissue_from_rgb
# ---------------------------------------------------------------------------

def test_segment_tissue_from_rgb_recovers_central_blob():
    image = _synthetic_tissue_image()
    mask = segment_tissue_from_rgb(image, sigma=2, close_radius=3, min_hole_area=100, min_object_area=100)
    assert mask.dtype == bool
    assert mask.shape == image.shape[:2]
    assert mask[100, 100]  # blob center is tissue
    assert not mask[5, 5]  # far corner is background
    # recovered area should be in the right ballpark of the true 80x80=6400px blob
    assert 4000 < mask.sum() < 10000


def test_segment_tissue_from_rgb_uniform_image_gives_empty_mask():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    mask = segment_tissue_from_rgb(image, sigma=2, close_radius=3, min_hole_area=100, min_object_area=100)
    assert not mask.any()


# ---------------------------------------------------------------------------
# get_hires_image
# ---------------------------------------------------------------------------

def test_get_hires_image_extracts_image_and_scale():
    image = _synthetic_tissue_image()
    adata = _adata_with_image(image=image, pixel_scale=0.17)
    got_image, got_scale = get_hires_image(adata, "sample1")
    np.testing.assert_array_equal(got_image, image)
    assert got_scale == pytest.approx(0.17)


def test_get_hires_image_respects_image_key():
    adata = _adata_with_image()
    lowres = np.zeros((10, 10, 3), dtype=np.uint8)
    adata.uns["spatial"]["sample1"]["images"]["lowres"] = lowres
    adata.uns["spatial"]["sample1"]["scalefactors"]["tissue_lowres_scalef"] = 0.05

    got_image, got_scale = get_hires_image(adata, "sample1", image_key="lowres")
    np.testing.assert_array_equal(got_image, lowres)
    assert got_scale == pytest.approx(0.05)


def test_get_hires_image_missing_uns_spatial_raises_clear_error():
    """The exact STOmics/Stereo-seq case: no adata.uns["spatial"] at all,
    since that platform's reader never embeds an image in the AnnData."""
    adata = anndata.AnnData(X=np.zeros((3, 2)), obs=pd.DataFrame(index=["a", "b", "c"]))
    with pytest.raises(KeyError, match="only work for platforms"):
        get_hires_image(adata, "sample1")


def test_get_hires_image_missing_library_id_raises_clear_error():
    adata = _adata_with_image(library_id="sample1")
    with pytest.raises(KeyError, match="library_id='other_sample' not found"):
        get_hires_image(adata, "other_sample")


def test_get_hires_image_missing_image_key_raises_clear_error():
    adata = _adata_with_image(library_id="sample1")
    with pytest.raises(KeyError, match="image_key='lowres' not found"):
        get_hires_image(adata, "sample1", image_key="lowres")


# ---------------------------------------------------------------------------
# get_tissue_mask
# ---------------------------------------------------------------------------

def test_get_tissue_mask_combines_extraction_and_segmentation():
    image = _synthetic_tissue_image()
    adata = _adata_with_image(image=image, pixel_scale=0.2)
    mask, pixel_scale = get_tissue_mask(
        adata, "sample1", sigma=2, close_radius=3, min_hole_area=100, min_object_area=100
    )
    assert mask.shape == image.shape[:2]
    assert mask[100, 100]
    assert pixel_scale == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# segment_tissue_from_ssdna
# ---------------------------------------------------------------------------

def _synthetic_ssdna(size=400):
    """Dark background (20), a dim ring (80) around a bright core (200) --
    tissue is brighter than background in a fluorescence stain."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    image = np.full((size, size), 20, dtype=np.uint8)
    image[r <= 150] = 80
    image[r <= 80] = 200
    return image, r


def test_segment_tissue_from_ssdna_otsu_keeps_bright_region_at_downscaled_resolution():
    image, r = _synthetic_ssdna()
    image[r <= 150] = 200  # two levels only: background vs. tissue
    mask, factor = segment_tissue_from_ssdna(image, target_size=100, close_radius=2)
    assert factor == 4
    assert mask.shape == (100, 100)
    # upsampled back, the mask matches the tissue disk up to a mask pixel
    native = np.repeat(np.repeat(mask, factor, 0), factor, 1)
    assert native[r <= 140].all()
    assert not native[r >= 160].any()


def test_segment_tissue_from_ssdna_multiotsu_top_keeps_only_brightest_class():
    image, r = _synthetic_ssdna()
    mask, factor = segment_tissue_from_ssdna(image, threshold="multiotsu_top", target_size=100, close_radius=2)
    native = np.repeat(np.repeat(mask, factor, 0), factor, 1)
    assert native[r <= 70].all()
    assert not native[r >= 90].any()


def test_segment_tissue_from_ssdna_fills_holes():
    image, r = _synthetic_ssdna()
    image[r <= 30] = 20  # dark hole in the middle of the tissue
    filled, factor = segment_tissue_from_ssdna(image, target_size=100, close_radius=1)
    holey, _ = segment_tissue_from_ssdna(image, target_size=100, close_radius=1, fill_holes=False)
    assert filled[50, 50] and not holey[50, 50]


def test_segment_tissue_from_ssdna_rejects_unknown_threshold():
    image, _ = _synthetic_ssdna()
    with pytest.raises(ValueError, match="threshold"):
        segment_tissue_from_ssdna(image, threshold="triangle")

