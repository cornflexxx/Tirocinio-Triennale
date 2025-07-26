import argparse
import os
import time
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="Generate/save float datasets as CSV or binary")
parser.add_argument("--outdir", type=str, default="../datasets", help="Output folder")
parser.add_argument("--size", type=int, default=1000, help="Number of elements per file")
parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Data type")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
parser.add_argument("--dist", type=str, default="uniform", choices=["uniform", "normal", "exponential", "randomwalk"], help="Distribution")
parser.add_argument("--files", type=int, default=1, help="Number of datasets to generate")
parser.add_argument("--mean", type=float, default=0.0, help="Mean (for normal)")
parser.add_argument("--std", type=float, default=1.0, help="Std (for normal)")
parser.add_argument("--low", type=float, default=0.0, help="Low (for uniform)")
parser.add_argument("--high", type=float, default=1.0, help="High (for uniform)")
parser.add_argument("--scale", type=float, default=1.0, help="Scale (for exponential)")
parser.add_argument("--smooth", action="store_true", help="Apply smoothing")
parser.add_argument("--kernel-size", type=int, default=5, help="Kernel size for smoothing")
parser.add_argument("--format", type=str, default="csv", choices=["csv", "bin"], help="Output format")
parser.add_argument("--rw-mean", type=float, default=0.0, help="Mean step for random walk")
parser.add_argument("--rw-std", type=float, default=1.0, help="Std step for random walk")

args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

dtype = torch.float32 if args.dtype == "float32" else torch.float64
device = torch.device(args.device)

for idx in range(args.files):
    if args.dist == "uniform":
        data = torch.empty(args.size, dtype=dtype, device=device).uniform_(args.low, args.high)
    elif args.dist == "normal":
        data = torch.empty(args.size, dtype=dtype, device=device).normal_(args.mean, args.std)
    elif args.dist == "exponential":
        data = torch.empty(args.size, dtype=dtype, device=device).exponential_(args.scale)
    elif args.dist == "randomwalk":
        steps = torch.empty(args.size, dtype =dtype, device=device).normal_(args.rw_mean, args.rw_std)
        data = torch.cumsum(steps, dim= 0)
    else: 
        raise ValueError("Unknown distribution")

    if args.smooth and args.kernel_size > 1 and args.kernel_size % 2 == 1:
        kernel = torch.ones(1, 1, args.kernel_size, device=device) / args.kernel_size
        x = data.unsqueeze(0).unsqueeze(0)
        data = F.conv1d(x, kernel, padding=args.kernel_size // 2).squeeze()

    data_np = data.cpu().numpy()

    filetype = args.format
    fname_base = f"{args.dist}_smooth-{args.smooth}_idx-{idx:03d}"
    outpath = os.path.join(args.outdir, f"{fname_base}.{filetype}")
    suffix = 0
    orig_outpath = outpath
    while os.path.exists(outpath):
        suffix += 1
        outpath = os.path.join(args.outdir, f"{fname_base}_{suffix}.{filetype}")
    if filetype == "csv":
        pd.DataFrame(data_np, columns=["value"]).to_csv(outpath, index=False)
    else:          
        data_np.tofile(outpath)

    size_mb = os.path.getsize(outpath) / (1024 * 1024)
    print(f"Saved: {outpath} ({size_mb:.2f} MB)")
