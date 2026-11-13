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
`segment_tissue_from_ssdna` segments those images once loaded (see
`anndata_api.quantify_diffusion_from_raw`, which finds them in a SAW output
folder).
"""
import numpy as np

__all__ = ["get_hires_image", "segment_tissue_from_rgb", "segment_tissue_from_he", "segment_tissue_from_ssdna",
           "get_tissue_mask"]


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

    Only works for platforms whose reader embeds an image directly in the
    AnnData this way -- e.g. 10x Visium. Platforms that ship tissue
    images/masks as separate files instead (e.g. STOmics/Stereo-seq) have
    no `adata.uns["spatial"]` at all and will raise a `KeyError` here --
    `anndata_api.quantify_diffusion_from_raw` reads those images straight
    from each pipeline's own output folder instead.

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
    if "spatial" not in adata.uns:
        raise KeyError(
            "adata.uns['spatial'] not found. get_hires_image/get_tissue_mask only work for platforms "
            "whose reader embeds an image directly in the AnnData (the scanpy/squidpy convention, e.g. "
            "10x Visium). If this sample stores its tissue mask/image as a separate file instead (e.g. "
            "STOmics/Stereo-seq), use anndata_api.quantify_diffusion_from_raw, which reads the "
            "image straight from the pipeline's output folder."
        )
    if library_id not in adata.uns["spatial"]:
        raise KeyError(
            f"library_id={library_id!r} not found in adata.uns['spatial']. "
            f"Available: {list(adata.uns['spatial'])}."
        )
    spatial_meta = adata.uns["spatial"][library_id]
    if image_key not in spatial_meta["images"]:
        raise KeyError(
            f"image_key={image_key!r} not found in adata.uns['spatial'][{library_id!r}]['images']. "
            f"Available: {list(spatial_meta['images'])}."
        )
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


def segment_tissue_from_he(image, pixel_size_um, low_fraction=0.25, sigma_um=10.0, edge_sigma_um=2.0, close_um=10.0,
                           min_hole_area_um2=5000.0, min_object_area_um2=20000.0, frame=0.05,
                           background_quantile=0.99, background_margin_um=50.0):
    """Tissue mask of an H&E image from its optical density relative to the
    slide background, grown from clearly stained tissue into faint tissue.

    1. Background colour: median RGB of the image frame (outer `frame` of each
       side), leaving out the darker half of a blurred-grey Otsu split (tissue
       reaching into the frame).
    2. Optical density OD = sum over RGB of max(0, -log10(I / background)),
       capped at 1 (pen marks, frame lines), Gaussian-blurred twice: by
       `sigma_um` to find and connect tissue (steps 3-4), and by
       `edge_sigma_um` to place its edge (step 5), so the heavy blur does not
       push the edge outwards.
    3. Seeds: OD above its Otsu threshold, not touching the image border.
    4. Growth: the seeds grow through every connected pixel with OD above
       max(`low_fraction` x Otsu, the `background_quantile` of the frame
       background), where the frame background leaves out pixels within
       `background_margin_um` of seed-dense structures (tissue, frame lines).
       Faint tissue joined to stained tissue is included; glass is not, so
       holes and tears inside the tissue stay open.
    5. Edge: the grown region is cut back to pixels whose lightly blurred OD
       is above the same growth threshold.
    6. Clean-up: closing (`close_um`), holes below `min_hole_area_um2` filled,
       objects below `min_object_area_um2` removed, nothing touching the
       image border.

    Unlike `segment_tissue_from_rgb` (one Otsu threshold on brightness) this
    keeps pale tissue whose brightness is close to the glass (e.g. pale
    stroma, weakly stained cores) and does not fill real holes.

    Parameters
    ----------
    image : np.ndarray
        RGB image (uint8 or float in [0, 1]).
    pixel_size_um : float
        Image pixel size (um); all size parameters are in um.
    low_fraction : float, default 0.25
        Growth threshold as a fraction of the Otsu (seed) threshold; 1 turns
        growth off. Raise it for very pale sections in stained haze, where
        growth leaks into the background.
    sigma_um : float, default 10
        Blur sigma (um) for finding and connecting tissue.
    edge_sigma_um : float, default 2
        Blur sigma (um) for placing the tissue edge.
    close_um : float, default 10
        Closing-disk radius (um).
    min_hole_area_um2 : float, default 5000
        Holes smaller than this are filled (~80 um across).
    min_object_area_um2 : float, default 20000
        Objects smaller than this are removed.
    frame : float, default 0.05
        Width of the image frame used for the background, as a fraction of
        each side.
    background_quantile : float, default 0.99
        Quantile of the frame background's OD that the growth threshold must
        exceed.
    background_margin_um : float, default 50
        Frame pixels within this distance of seed-dense structures do not
        count as background.

    Returns
    -------
    np.ndarray of bool
        Tissue mask, same shape as `image`'s first two dimensions.
    """
    _require_skimage()
    from scipy import ndimage
    from skimage.color import rgb2gray
    from skimage.filters import gaussian, threshold_otsu
    from skimage.morphology import binary_closing, disk, remove_small_holes, remove_small_objects
    from skimage.segmentation import clear_border

    rgb = np.asarray(image)[..., :3].astype(float)
    if np.asarray(image).dtype == np.uint8:
        rgb /= 255.0
    um = float(pixel_size_um)
    h, w = rgb.shape[:2]
    in_frame = np.zeros((h, w), dtype=bool)
    fh, fw = max(1, int(frame * h)), max(1, int(frame * w))
    in_frame[:fh] = in_frame[-fh:] = True
    in_frame[:, :fw] = in_frame[:, -fw:] = True

    grey = gaussian(rgb2gray(rgb), sigma=sigma_um / um)
    if np.ptp(grey) == 0:
        return np.zeros((h, w), dtype=bool)
    bright = grey >= threshold_otsu(grey)
    bg_pixels = in_frame & bright if (in_frame & bright).any() else in_frame
    background = np.median(rgb[bg_pixels], axis=0)

    od_raw = np.clip(-np.log10(np.clip(rgb, 1e-3, None) / np.clip(background, 1e-3, None)), 0, None).sum(-1)
    od_raw = np.minimum(od_raw, 1.0)
    od = gaussian(od_raw, sigma=sigma_um / um)
    od_edge = gaussian(od_raw, sigma=edge_sigma_um / um) if edge_sigma_um > 0 else od_raw
    if np.ptp(od) == 0:
        return np.zeros((h, w), dtype=bool)
    high = threshold_otsu(od[::4, ::4])
    dense = od > high
    near_dense = ndimage.distance_transform_edt(~dense) * um < background_margin_um
    frame_od = od[in_frame & ~near_dense]
    low = low_fraction * high
    if frame_od.size:
        low = max(low, float(np.quantile(frame_od, background_quantile)))
    low = min(low, high)

    mask = ndimage.binary_propagation(clear_border(dense), mask=od > low)
    mask = clear_border(mask) & (od_edge > low)
    mask = binary_closing(mask, disk(max(1, int(round(close_um / um)))))
    mask = remove_small_holes(mask, area_threshold=max(1, int(min_hole_area_um2 / um**2)))
    mask = remove_small_objects(mask, min_size=max(1, int(min_object_area_um2 / um**2)))
    return mask


def segment_tissue_from_ssdna(image, threshold="otsu", target_size=2000, close_radius=6, fill_holes=True):
    """Tissue-vs-background segmentation for a single-channel fluorescence
    nucleic-acid stain (e.g. STOmics/Stereo-seq's `*_ssDNA_regist.tif`):
    block-mean downscale -> global threshold (tissue is *brighter* than the
    background here, unlike H&E) -> morphological closing -> hole filling.

    Segmentation runs on the downscaled image only -- a native ssDNA image is
    ~23k x 23k pixels, far too large (and far too textured) to threshold
    usefully at native resolution. The returned mask stays at that
    downscaled resolution; `factor` maps it back (native pixel `(r, c)` lies
    in mask pixel `(r // factor, c // factor)`).

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image.
    threshold : {"otsu", "multiotsu_top"}, default "otsu"
        "otsu": single Otsu threshold. "multiotsu_top": 3-class multi-Otsu,
        keeping only the brightest class -- for images where a dim middle
        class is background (e.g. uneven/bright chip background) rather than
        tissue.
    target_size : int, default 2000
        Approximate long-edge size (pixels) of the downscaled working image;
        the downscale factor is `max(1, round(max(image.shape) / target_size))`.
    close_radius : int, default 6
        Radius (downscaled pixels) of the morphological closing disk.
    fill_holes : bool, default True
        Fill enclosed holes after closing.

    Returns
    -------
    mask : np.ndarray of bool
        Tissue mask at the downscaled resolution.
    factor : int
        Integer downscale factor (native pixels per mask pixel, per axis).
    """
    _require_skimage()
    from scipy.ndimage import binary_fill_holes
    from skimage.filters import threshold_multiotsu, threshold_otsu
    from skimage.morphology import binary_closing, disk
    from skimage.transform import downscale_local_mean

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got shape {image.shape}.")
    factor = max(1, int(round(max(image.shape) / target_size)))
    small = downscale_local_mean(image, (factor, factor)).astype(float)

    if threshold == "otsu":
        mask = small > threshold_otsu(small)
    elif threshold == "multiotsu_top":
        mask = small > threshold_multiotsu(small, classes=3)[-1]
    else:
        raise ValueError(f"threshold must be 'otsu' or 'multiotsu_top', got {threshold!r}.")

    mask = binary_closing(mask, disk(close_radius))
    if fill_holes:
        mask = binary_fill_holes(mask)
    return mask, factor


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
