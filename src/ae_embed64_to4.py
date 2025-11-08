"""
This script trains a convolutional AutoEncoder to compress 64-band embedding into 4 bands
and export the 4-band bottleneck as a GeoTIFF with the same georeferencing.
"""

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine

def gaussian_kernel_1d(kernel_size: int, sigma: float, device):
    coords = torch.arange(kernel_size, device=device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g

def create_ssim_window(channels, window_size=11, sigma=1.5, device="cpu"):
    _1d = gaussian_kernel_1d(window_size, sigma, device)
    _2d = (_1d[:, None] * _1d[None, :]).unsqueeze(0).unsqueeze(0)
    window = _2d.repeat(channels, 1, 1, 1) 
    return window

def ssim_torch(x, y, window, C1=0.01**2, C2=0.03**2):
    C = x.size(1)
    padding = window.size(-1) // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=C)
    mu_y = F.conv2d(y, window, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()

def scc_torch(x, y, eps=1e-8):
    N, C, H, W = x.shape
    x_flat = x.reshape(N, C, -1)
    y_flat = y.reshape(N, C, -1)
    x_mean = x_flat.mean(dim=-1, keepdim=True)
    y_mean = y_flat.mean(dim=-1, keepdim=True)
    x_c = x_flat - x_mean
    y_c = y_flat - y_mean
    num = (x_c * y_c).sum(dim=-1)
    den = torch.sqrt((x_c ** 2).sum(dim=-1) * (y_c ** 2).sum(dim=-1) + eps)
    corr = num / (den + eps)  # (N,C)
    return corr.mean()

def hann2d(h, w, device="cpu", eps=0.05):
    wy = torch.hann_window(h, periodic=False, device=device).unsqueeze(1)
    wx = torch.hann_window(w, periodic=False, device=device).unsqueeze(0)
    w2d = (wy * wx)  # (h,w) in [0,1]
    if eps is not None and eps > 0:
        w2d = eps + (1.0 - eps) * w2d
    return w2d

def to_numpy(t):
    return t.detach().cpu().numpy()

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3, stride=1, use_bn=True):
        super().__init__()
        pad = ks // 2
        layers = [
            nn.Conv2d(
                in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad,
                bias=not use_bn, padding_mode='reflect' 
            )
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, ks=3, stride=1, use_bn=True):
        super().__init__()
        self.e1 = ConvBlock(64, 34, ks, stride, use_bn)
        self.e2 = ConvBlock(34, 16, ks, 1, use_bn)
        self.e3 = ConvBlock(16, 8, ks, 1, use_bn)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        z = self.conv4(x)
        return z 

class Decoder(nn.Module):
    def __init__(self, ks=3, use_bn=True):
        super().__init__()
        self.d1 = ConvBlock(4, 8, ks, 1, use_bn)
        self.d2 = ConvBlock(8, 16, ks, 1, use_bn)
        self.d3 = ConvBlock(16, 34, ks, 1, use_bn)
        self.conv_out = nn.Conv2d(34, 64, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, z):
        x = self.d1(z)
        x = self.d2(x)
        x = self.d3(x)
        out = self.conv_out(x)
        return out 

class AE(nn.Module):
    def __init__(self, ks=3, stride=1, use_bn=True):
        super().__init__()
        self.encoder = Encoder(ks=ks, stride=stride, use_bn=use_bn)
        self.decoder = Decoder(ks=ks, use_bn=use_bn)

    def forward(self, x):
        z = self.encoder(x)
        rec = self.decoder(z)
        return z, rec

class GeoTiffPatches(Dataset):
    def __init__(self, arr, patch=128, overlap=16):
        """
        arr: np.ndarray (C,H,W) float32 standardized
        """
        self.arr = arr
        self.C, self.H, self.W = arr.shape
        self.patch = patch
        self.overlap = overlap
        self.grid = []
        stride = patch - overlap
        for y in range(0, self.H, stride):
            for x in range(0, self.W, stride):
                y0 = y
                x0 = x
                y1 = min(y0 + patch, self.H)
                x1 = min(x0 + patch, self.W)
                y0 = max(0, y1 - patch)
                x0 = max(0, x1 - patch)
                self.grid.append((y0, y1, x0, x1))

    def __len__(self):
        return len(self.grid)

    def __getitem__(self, idx):
        y0, y1, x0, x1 = self.grid[idx]
        patch = self.arr[:, y0:y1, x0:x1] 
        return torch.from_numpy(patch)

def write_geotiff_like(src_profile, out_path, data_4chw):
    """
    data_4chw: (4, H, W) numpy float32
    """
    prof = src_profile.copy()
    prof.update({
        "count": 4,
        "dtype": rasterio.float32
    })
    with rasterio.open(out_path, "w", **prof) as dst:
        for i in range(4):
            dst.write(data_4chw[i], i + 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_tif", required=True, help="Path to input 64-band embedding GeoTIFF")
    ap.add_argument("--out_tif", required=True, help="Path to output 4-band bottleneck GeoTIFF")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ks", type=int, default=3, help="Conv kernel size")
    ap.add_argument("--stride", type=int, default=1, help="Encoder first block stride (1 keeps HxW)")
    ap.add_argument("--bn", type=int, default=1, help="Use BatchNorm (1) or not (0)")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--edge_eps", type=float, default=0.05,
                    help="Floor added to Hann window so borders keep non-zero weight (e.g., 0.05).")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--w_mse", type=float, default=1.0)
    ap.add_argument("--w_ssim", type=float, default=0.5)
    ap.add_argument("--w_scc", type=float, default=0.5)
    ap.add_argument("--w_cos", type=float, default=0.5)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with rasterio.open(args.in_tif) as src:
        profile = src.profile
        bands = src.count
        assert bands == 64, f"Expected 64 bands, got {bands}"
        arr = src.read(out_dtype="float32")

    mean = arr.reshape(arr.shape[0], -1).mean(axis=1, keepdims=True)
    std = arr.reshape(arr.shape[0], -1).std(axis=1, keepdims=True) + 1e-6
    arr_std = ((arr.reshape(arr.shape[0], -1) - mean) / std).reshape(arr.shape)

    ds = GeoTiffPatches(arr_std, patch=args.patch, overlap=args.overlap)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AE(ks=args.ks, stride=args.stride, use_bn=bool(args.bn)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ssim_window = create_ssim_window(channels=64, window_size=11, sigma=1.5, device=device)

    model.train()
    stop_threshold = 0.02 
    stopped = False

    for epoch in range(1, args.epochs + 1):
        total = 0.0
        for batch in dl:
            x = batch.to(device)  
            opt.zero_grad()
            z, rec = model(x)

            mse = F.mse_loss(rec, x)

            ssim_val = ssim_torch(rec, x, ssim_window)  
            l_ssim = 1.0 - ssim_val

            scc_val = scc_torch(rec, x)  
            l_scc = 1.0 - scc_val

            rec_flat = rec.flatten(2)  
            x_flat = x.flatten(2)
            cos_sim = F.cosine_similarity(rec_flat, x_flat, dim=1).mean()  
            l_cos = 1.0 - cos_sim

            loss = args.w_mse * mse + args.w_ssim * l_ssim + args.w_scc * l_scc + args.w_cos * l_cos
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)

            if loss.item() < stop_threshold:
                print(f"[Early stop] epoch={epoch} batch_loss={loss.item():.6f} < {stop_threshold}")
                stopped = True
                break

        avg_loss = total / len(ds)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={avg_loss:.6f} | mse={mse.item():.4f} "
              f"ssim={ssim_val.item():.4f} scc={scc_val.item():.4f} cos={cos_sim.item():.4f}")
        if stopped or (avg_loss < stop_threshold):
            print(f"Early stopping at epoch {epoch}, avg_loss={avg_loss:.6f}")
            break

    model.eval()
    with torch.no_grad():
        C, H, W = arr_std.shape
        out4   = np.zeros((4, H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)

        stride = args.patch - args.overlap

        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y0, x0 = y, x
                y1 = min(y0 + args.patch, H)
                x1 = min(x0 + args.patch, W)
                y0 = max(0, y1 - args.patch)
                x0 = max(0, x1 - args.patch)

                patch = arr_std[:, y0:y1, x0:x1]                 
                ten   = torch.from_numpy(patch).unsqueeze(0).to(device)
                z, _  = model(ten)                                
                z_np  = z.squeeze(0).cpu().numpy()               

                w2d   = hann2d(y1-y0, x1-x0, device=device, eps=args.edge_eps)
                w2dnp = to_numpy(w2d).astype(np.float32)

                out4[:, y0:y1, x0:x1] += z_np * w2dnp[None, :, :]
                weight[y0:y1, x0:x1]   += w2dnp

        out4 /= np.clip(weight[None, :, :], 1e-6, None)

    mask_w = (weight > (args.edge_eps * 0.5)).astype(np.float32)  
    flat   = out4.reshape(4, -1)                                 
    mflat  = mask_w.reshape(-1)                                    
    denom  = mflat.sum() + 1e-6
    z_mean = ((flat * mflat[None, :]).sum(axis=1, keepdims=True) / denom)     
    z_std  = np.sqrt((((flat - z_mean) ** 2) * mflat[None, :]).sum(axis=1, keepdims=True) / denom + 1e-6)
    out4_norm01 = ((flat - z_mean) / (2.5 * z_std))
    out4_norm01 = out4_norm01.reshape(out4.shape)                 

    write_geotiff_like(profile, args.out_tif, out4_norm01)
    print(f"[OK] Saved 4-band bottleneck to: {args.out_tif}")

if __name__ == "__main__":
    main()
