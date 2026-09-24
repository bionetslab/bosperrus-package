"""Image-derived tissue masks for image-backed spatial-transcriptomics data.

Requires the optional `image-masks` extra (`pip install bosperrus[image-masks]`,
which pulls in `scikit-image` and `anndata`). Like `anndata_api.py` and
`distances.distance_to_alpha_shape`, heavy imports are lazy (inside the
functions themselves), so the rest of `bosperrus` -- and this module's own
presence in `bosperrus.__all__` -- stays importable without them installed.

`get_hires_image` only covers the scanpy/squidpy AnnData convention
(`adata.uns["spatial"][library_id]`), used broadly across platforms whose
readers embed an image directly in the AnnData -- e.g. 10x Visium. Platforms
that ship tissue images/masks as separate files with their own project-
specific naming convention (e.g. STOmics/Stereo-seq) aren't covered here;
that loading logic is inherently tied to one project's file layout, not a
general AnnData convention, so it belongs with that project's own code.
"""
import numpy as np

__all__ = ["get_hires_image", "segment_tissue_from_rgb", "get_tissue_mask"]


def _require_skimage():
    try:
        import skimage  # noqa: F401
    except ImportError:
        raise ImportError(
            "scikit-image is required for this function. "
            "Install it with: pip install scikit-image or pip install bosperrus[image-masks]"
        )


def _require_anndata():
    try:
        import anndata  # noqa: F401
    except ImportError:
        raise ImportError(
            "anndata is required for this function. "
            "Install it with: pip install anndata or pip install bosperrus[anndata]"
        )


def get_hires_image(adata, library_id, image_key="hires"):
    """Extract an AnnData's own embedded image and its pixel scale factor
    (scanpy/squidpy convention: `adata.uns["spatial"][library_id]`).

    Parameters
    ----------
    adata : AnnData
    library_id : str
        Key into `adata.uns["spatial"]` (e.g. the sample name).
    image_key : str, default "hires"
        Which embedded image to use (e.g. "hires" or "lowres") -- looked up as
        `adata.uns["spatial"][library_id]["images"][image_key]`, alongside the
        matching `adata.uns["spatial"][library_id]["scalefactors"]["tissue_{image_key}_scalef"]`.

    Returns
    -------
    image : np.ndarray
    pixel_scale : float
        Converts a spot's native `adata.obsm["spatial"]` (full-resolution
        pixel) coordinate into this image's own pixel coordinate via
        `image_xy = spatial_xy * pixel_scale` -- the same factor platforms
        following this convention use to align spots to a non-fullres image.
    """
    _require_anndata()
    spatial_meta = adata.uns["spatial"][library_id]
    image = spatial_meta["images"][image_key]
    pixel_scale = float(spatial_meta["scalefactors"][f"tissue_{image_key}_scalef"])
    return image, pixel_scale


def segment_tissue_from_rgb(image, sigma=8, close_radius=10, min_hole_area=50000, min_object_area=3000):
    """Simple, uniform tissue-vs-background segmentation for an RGB
    histology image (e.g. H&E): grayscale -> heavy Gaussian blur -> Otsu
    threshold (tissue is darker than the white/light slide background) ->
    drop anything touching the image border -> morphological closing +
    small-hole-filling + small-object removal, to turn the raw threshold
    into a handful of solid tissue blobs instead of a speckled
    "nuclei only" mask (a single global Otsu on the *unblurred* grayscale
    image tends to pick out only the darkest nuclei-dense foci, not the bulk
    tissue outline, since histology images have a lot of internal texture --
    the blur washes that out first).

    One fixed parameter set is not guaranteed to suit every image -- tune
    `sigma`/`close_radius`/`min_hole_area`/`min_object_area` if the result
    merges nearby tissue pieces that should stay separate, or drops small
    genuine fragments (e.g. a TMA's individual cores).

    Parameters
    ----------
    image : np.ndarray
        RGB image array.
    sigma : float, default 8
        Gaussian blur sigma (pixels) applied before Otsu thresholding.
    close_radius : int, default 10
        Radius (pixels) of the morphological closing disk.
    min_hole_area : int, default 50000
        Holes up to this area (pixels) are filled in.
    min_object_area : int, default 3000
        Connected components smaller than this (pixels) are removed.

    Returns
    -------
    np.ndarray of bool
        Tissue mask, same shape as `image`'s first two dimensions.
    """
    _require_skimage()
    from skimage.color import rgb2gray
    from skimage.filters import gaussian, threshold_otsu
    from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects
    from skimage.segmentation import clear_border

    gray = rgb2gray(image)
    blurred = gaussian(gray, sigma=sigma)
    mask = blurred < threshold_otsu(blurred)
    mask = clear_border(mask)
    mask = binary_closing(mask, disk(close_radius))
    mask = remove_small_holes(mask, area_threshold=min_hole_area)
    mask = remove_small_objects(mask, min_size=min_object_area)
    return mask


def get_tissue_mask(adata, library_id, image_key="hires", **segment_kwargs):
    """`get_hires_image` + `segment_tissue_from_rgb` in one call.

    Returns
    -------
    mask : np.ndarray of bool
    pixel_scale : float
        See `get_hires_image` -- shared by both `mask` and the source image.
    """
    image, pixel_scale = get_hires_image(adata, library_id, image_key=image_key)
    mask = segment_tissue_from_rgb(image, **segment_kwargs)
    return mask, pixel_scale
