"""
This script is used to run model across a regional dataset,
compute global and cloud-covered MSE/SSIM metrics with dynamic SSIM window sizes,
and save reconstructed S2 outputs as GeoTIFF. Uses default DIP epoch_steps=1000 and Trainer max_epochs=4.

To run, from project root:
    python -m src.run_LitDIPrsps \
      --base_dir ".../your/image/folder/" \
      --cloud_dir ".../cloud masks/folder/" \
      --date_start YYYYMMDD \
      --date_end   YYYYMMDD \
      --embedding4_path ".../your/embeddings.tif" \
      --cluster_path ".../your/cluters.tif" \
      --cluster_shape_weight weight \
      --edge_tau weight \
      --edge_dilate number \
      --filter_file ".../your clear sky file.txt" \
      --output_dir ".../output/" \
      --gpus n
"""

import os
import argparse
import numpy as np
import csv
from tqdm import tqdm
import pytorch_lightning as pl
import rasterio
from rasterio.transform import from_origin
from skimage.metrics import mean_squared_error, structural_similarity as ssim
from src.LitDIPrsps import LitDIPrsps
from src.RegionalDataset import RegionalDataset

def compute_global_ssim(im1, im2):
    h, w, c = im1.shape
    win = min(7, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    data_range = im1.max() - im1.min()
    vals = []
    for b in range(c):
        vals.append(ssim(im1[..., b], im2[..., b], win_size=win, data_range=data_range))
    return float(np.mean(vals))

def compute_cloud_ssim(im1, im2, mask):
    h, w, c = im1.shape
    win = min(7, h, w)
    if win % 2 == 0:
        win -= 1
    win = max(win, 3)
    data_range = im1.max() - im1.min()
    cloud = ~mask.astype(bool)
    vals = []
    for b in range(c):
        _, mapb = ssim(im1[..., b], im2[..., b], full=True, win_size=win, data_range=data_range)
        vals.append(np.nanmean(mapb[cloud]))
    return float(np.mean(vals))

def main():
    parser = argparse.ArgumentParser(description="Run DIP cloud removal on dataset")
    parser.add_argument('--base_dir',    type=str, required=True)
    parser.add_argument('--cloud_dir',   type=str, required=True)
    parser.add_argument('--filter_file', type=str, default='clear_sky.txt')
    parser.add_argument('--date_start',  type=str, default='20200101')
    parser.add_argument('--date_end',    type=str, default='20201231')
    parser.add_argument('--embedding4_path', type=str, required=True,
                        help='Path to a 4-band embedding GeoTIFF aligned to S2 tiles')
    parser.add_argument('--cluster_path', type=str, default=None,
                        help='Optional single-band cluster raster (values 1..K) aligned to S2')
    parser.add_argument('--cluster_weight', type=float, default=0.0,
                        help='>0 to enable cluster distribution regularization')
    parser.add_argument('--cluster_shape_weight', type=float, default=0.0,
                        help='>0 enables shape prior (within-cluster TV + boundary alignment)')
    parser.add_argument('--edge_tau', type=float, default=0.03, help='Boundary gradient floor')
    parser.add_argument('--edge_dilate', type=int, default=1, help='Boundary dilation in pixels')
    parser.add_argument('--output_dir',  type=str, default='predictions')
    parser.add_argument('--gpus',        type=int, default=1)
    args = parser.parse_args()

    ds = RegionalDataset_S2only(
        base_dir=args.base_dir,
        cloud_dir=args.cloud_dir,
        date_range=(args.date_start, args.date_end),
        filter_file=args.filter_file,
        shuffle=False,
        db_sar = False
    )
    with rasterio.open(args.embedding4_path) as _emb:
        _emb_arr = _emb.read(out_dtype='float32')  
    emb4 = np.moveaxis(_emb_arr, 0, -1)          
    cluster_arr = None
    if args.cluster_path is not None:
        with rasterio.open(args.cluster_path) as _clu:
            cluster_arr = _clu.read(1).astype(np.int64)
    os.makedirs(args.output_dir, exist_ok=True)

    csv_path = os.path.join(args.output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['date','roi','cloud_idx','rmse_global','ssim_global','rmse_cloud','ssim_cloud'])

    total = len(ds)
    print(f"Processing {total} samples...")
    transform = from_origin(0,0,1,1)
    crs = None

    for idx in tqdm(range(total)):
        nm = len(ds.cloud_files)
        date_idx, cloud_idx = divmod(idx, nm)
        s2_clear, mask = ds[idx]
        fill = 255.0 / ds.s2_scale
        s2_cor = s2_clear.copy()
        cloud_pixels = ~mask.astype(bool)
        for b in range(s2_cor.shape[2]):
            band = s2_cor[:,:,b]
            band[cloud_pixels] = fill
            s2_cor[:,:,b] = band

        model = LitDIPrsps(
            cluster_loss_weight=args.cluster_weight,
            shape_loss_weight=args.cluster_shape_weight,
            edge_tau=args.edge_tau,
            edge_dilate=args.edge_dilate
        )

        model.set_target([s2_cor, emb4])
        model.set_mask([mask.astype(bool), np.ones(mask.shape, bool)])
        if cluster_arr is not None:
            model.set_cluster(cluster_arr)
        pl.Trainer(max_epochs=4,
                   gpus=[0] if args.gpus>0 else None,
                   checkpoint_callback=False,
                   logger=False).fit(model)
        rec = model.output()[0]

        mse_global = mean_squared_error(s2_clear, rec)
        rmse_global = float(np.sqrt(mse_global))
        ssim_global = compute_global_ssim(s2_clear, rec)
        mse_cloud = (mean_squared_error(s2_clear[cloud_pixels], rec[cloud_pixels]) \
                     if cloud_pixels.any() else np.nan)
        rmse_cloud = (float(np.sqrt(mse_cloud)) if cloud_pixels.any() else np.nan)
        ssim_cloud = compute_cloud_ssim(s2_clear, rec, mask)

        date, roi = ds.used_dates[date_idx].split('/')
        tif_path = os.path.join(args.output_dir, f"{date}-{roi}-{cloud_idx}.tif")
        from pathlib import Path as _Path
        import warnings as _warnings

        _s2_dir = _Path(ds.base_dir) / date / roi / "S2"
        _s2_files = sorted(list(_s2_dir.glob("*.tif")) + list(_s2_dir.glob("*.tiff")))
        if _s2_files:
            with rasterio.open(_s2_files[0]) as _ref:
                _profile = _ref.profile.copy()
            _profile.update(height=rec.shape[0], width=rec.shape[1], count=rec.shape[2], dtype=rec.dtype)
            with rasterio.open(tif_path, "w", **_profile) as dst:
                dst.write(rec.transpose(2,0,1))
        else:
            with rasterio.open(tif_path, 'w', driver='GTiff', 
                               height=rec.shape[0], width=rec.shape[1],
                               count=rec.shape[2], dtype=rec.dtype,
                               crs=crs, transform=transform) as dst:
                dst.write(rec.transpose(2,0,1))
        with open(csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([date, roi, cloud_idx,
                        rmse_global, ssim_global,
                        rmse_cloud, ssim_cloud])
    print(f"Done. Outputs and metrics in {args.output_dir}")

if __name__ == '__main__':
    main()
