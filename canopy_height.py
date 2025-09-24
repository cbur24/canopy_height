import laspy
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
import requests
import tempfile
import os
import odc.geo
from odc.geo.xr import assign_crs

def laz_to_canopy_height(
    source,
    resolution=1.0,
    save_tif="chm.tif",
    download_chunk_size=1_048_576 #1MB
):
    """
    Convert a LiDAR .laz/.las file to a Canopy Height Model (CHM)
    using min/max binning.

    How canopy height is calculated:

    `DEM` = ground surface (lowest return in cell)
    `DSM` = top surface (highest return in cell)
    `CHM` = DSM - DEM

    Assumes
     - Ground = minimum elevation in cell
     - Top of canopy = maximum elevation in cell
     - We aren’t using the LAS classification flags (e.g., ground vs vegetation).

    Limitations
    - Dense canopy without ground penetration = DEM will be too high, CHM too low.
    - Isolated noisy points = DSM will be too high, CHM overestimated.
    - Buildings = counted as 'canopy' unless explicitly removed.

    
    Parameters
    ----------
    source : str
        Path to local .laz/.las file, or a URL to download.
    resolution : float, optional
        Output CHM pixel size in the same units as LiDAR coords.
    save_tif : str or None, optional
        If given, path to save CHM as a GeoTIFF.
    download_chunk_size : int, optional
        Chunk size (bytes) when downloading remote files.

    Returns
    -------
    chm_da : xarray.DataArray
        Canopy Height Model
    
    """
    if source.startswith("http"):
        tmpdir = tempfile.TemporaryDirectory()
        local_path = os.path.join(tmpdir.name, os.path.basename(source))

        with requests.get(source, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length", 0))
            downloaded = 0

            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=download_chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        percent = downloaded * 100 / total_size
                        print(f"\rDownload progress: {percent:.2f}% ({downloaded}/{total_size} bytes)", end="")

    else:
        local_path = source
        tmpdir = None

    # Load point cloud
    las = laspy.read(local_path)
    x, y, z = las.x, las.y, las.z

    # Grid size
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    nx = int(np.ceil((xmax - xmin) / resolution))
    ny = int(np.ceil((ymax - ymin) / resolution))

    # Grid indices
    ix = ((x - xmin) / resolution).astype(np.int32)
    iy = ((ymax - y) / resolution).astype(np.int32)  # flip y for raster coords
    flat_idx = iy * nx + ix

    # Initialise arrays
    dem_fill = np.full(nx * ny, np.inf, dtype=np.float32)
    dsm_fill = np.full(nx * ny, -np.inf, dtype=np.float32)

    # find min and max returns
    np.minimum.at(dem_fill, flat_idx, z)
    np.maximum.at(dsm_fill, flat_idx, z)

    # Replace inf with NaN
    dem_fill[dem_fill == np.inf] = np.nan
    dsm_fill[dsm_fill == -np.inf] = np.nan

    # Canopy height
    chm_flat = dsm_fill - dem_fill
    chm_flat[chm_flat < 0] = 0
    chm_grid = chm_flat.reshape((ny, nx))

    # Coordinates (centre of pixels) -  0.5 is in pixel units
    x_coords = xmin + (np.arange(nx) + 0.5) * resolution
    y_coords = ymax - (np.arange(ny) + 0.5) * resolution

    # Build DataArray
    chm_da = xr.DataArray(
        chm_grid,
        coords={"y": y_coords, "x": x_coords},
        dims=("y", "x"),
        name="canopy_height"
    )

    # assign ODC accessor and crs attrs
    crs = las.header.parse_crs()
    chm_da = assign_crs(chm_da, crs=crs)

    # Attach CRS as an added metadata attrs
    if crs is not None:
        chm_da.attrs["crs"] = crs.to_wkt()

    # Save to GeoTIFF if requested
    if save_tif is not None:
        chm_da.odc.write_cog(save_tif, overwite=True)
    
    # Cleanup tempdir if remote
    if tmpdir is not None:
        tmpdir.cleanup()

    return chm_da