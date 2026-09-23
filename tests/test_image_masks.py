import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skimage", reason="scikit-image not installed")
anndata = pytest.importorskip("anndata", reason="anndata not installed")

from bosperrus.image_masks import get_hires_image, segment_tissue_from_rgb, get_tissue_mask


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
