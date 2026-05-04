#!/usr/bin/env python3
import os
import re
import glob
import torch
import argparse
import math


def find_latest_checkpoint(path):
    checkpoints_dir = os.path.join(path, 'checkpoints')
    search_dir = checkpoints_dir if os.path.isdir(checkpoints_dir) else path
    pattern = os.path.join(search_dir, 'checkpoint_epoch_*.pyth')
    files = glob.glob(pattern)
    if not files:
        raise ValueError(f"No .pyth checkpoint files found in {search_dir}")
    epochs = [(int(re.search(r'checkpoint_epoch_(\d+)\.pyth', os.path.basename(f)).group(1)), f)
              for f in files if re.search(r'checkpoint_epoch_(\d+)\.pyth', os.path.basename(f))]
    _, latest_file = max(epochs, key=lambda x: x[0])
    return latest_file

@torch.no_grad()
def tsv_m_statevec(task_vectors, device="cpu"):
    sv_reduction = 1 / len(task_vectors)
    print("Computing SVD...")
    new_vector = {}
    for key in task_vectors[0]:
        new_vector[key] = {}
        for i, task_vector in enumerate(task_vectors):
            vec = task_vector[key].to(device)
            if (
                len(task_vector[key].shape) == 2
                and "text_projection" not in key
            ):
                u, s, v = torch.linalg.svd(vec, full_matrices=False)
                if i == 0:
                    print(f"Computed SVD for {key}...")
                    sum_u = torch.zeros_like(u, device=device)
                    sum_s = torch.zeros_like(s, device=device)
                    sum_v = torch.zeros_like(v, device=device)
                reduced_index_s = int(s.shape[0] * sv_reduction)
                # select only the first reduced_index_s columns of u and place them
                sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[
                    :, :reduced_index_s
                ]
                sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[
                    :reduced_index_s
                ]
                # select only the first reduced_index_s rows of v and place them
                sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[
                    :reduced_index_s, :
                ]
            else:
                if i == 0:
                    new_vector[key] = vec.clone()
                else:
                    new_vector[key] += (vec - new_vector[key]) / (i + 1)
        if len(task_vector[key].shape) == 2 and "text_projection" not in key:
            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            new_vector[key] = torch.linalg.multi_dot(
                (
                    u_u,
                    v_u,
                    torch.diag(sum_s),
                    u_v,
                    v_v,
                )
            )

    return new_vector

@torch.no_grad()
def iso_c_statevec(state_dicts, device='cpu'):
    merged = {}
    keys = state_dicts[0].keys()
    num_tv = len(state_dicts)
    avg = {k: sum(sd[k].to(device).float() for sd in state_dicts) / num_tv for k in keys}

    for k, w in avg.items():
        if w.dim() == 2:
            try:
                w *= num_tv
                U, S, Vh = torch.linalg.svd(w, full_matrices=False)
                S_mean = torch.ones_like(S) * S.mean()
                merged[k] = U @ torch.diag(S_mean) @ Vh
                continue
            except Exception as e:
                print(f"Iso-C failed on {k}, fallback to mean: {e}")
        merged[k] = w
    return merged


def merge_checkpoints(
    ckpt_paths,
    mode="mean",
    base_state=None,
    common_frac=0.8,
    alpha=1.0,
    return_delta=False,
    device='cpu',
):
    sd_list = []
    print(f"Loading {len(ckpt_paths)} checkpoints...")
    for p in ckpt_paths:
        if os.path.isdir(p):
            p = find_latest_checkpoint(p)
            print(f"Using checkpoint: {p}")
        ck = torch.load(p, map_location="cpu")
        sd = ck.get("model_state") or ck.get("state_dict") or ck
        if base_state:
            sd = {k: sd[k].float() - base_state[k].float() for k in sd if k in base_state}
        else:
            sd = {k: sd[k].float() for k in sd}
        sd_list.append(sd)

    # Merge
    if mode == "mean":
        delta = {k: sum(sd[k] for sd in sd_list) / len(sd_list) for k in sd_list[0]}
    elif mode == "iso-c":
        delta = iso_c_statevec(sd_list, device=device)
    elif mode == "tsv-m":
        delta = tsv_m_statevec(sd_list, device=device)
    else:
        raise ValueError(f"Unknown merge mode: {mode}")

    if return_delta:
        return delta

    # Apply alpha and add back to base weights
    if base_state:
        merged_final = {
            k: base_state[k].float() + alpha * delta.get(k, torch.zeros_like(base_state[k]))
            for k in base_state
        }
    else:
        raise ValueError("Base checkpoint (--base) is required to reconstruct full weights.")

    return merged_final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge model checkpoints (mean, iso-c)")
    parser.add_argument("--ckpts", nargs="+", required=True, help="List of checkpoint paths")
    parser.add_argument("--mode", choices=["mean", "iso-c", "tsv-m"], default="mean", help="Merging method")
    parser.add_argument("--base", type=str, required=True, help="Path to base checkpoint (w0)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Scaling factor α")
    parser.add_argument(
        "--save_delta",
        action="store_true",
        help="Save merged task vector only (base reconstruction deferred to eval).",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device for computation")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    base_ckpt = torch.load(args.base, map_location="cpu")
    base_state = base_ckpt.get("model_state") or base_ckpt.get("state_dict") or base_ckpt

    merged = merge_checkpoints(
        ckpt_paths=args.ckpts,
        mode=args.mode,
        base_state=base_state,
        alpha=args.alpha,
        return_delta=args.save_delta,
        device=args.device
    )

    if args.save_delta:
        filename = f"{args.mode}_delta.pyth"
    else:
        filename = f"{args.mode}_alpha{str(args.alpha).replace('.', '-')}.pyth"
    save_path = os.path.join(args.output_dir, filename)
    torch.save(
        {
            "model_state": merged,
            "is_delta": bool(args.save_delta),
            "base_checkpoint": args.base,
            "merge_mode": args.mode,
            "source_checkpoints": args.ckpts,
        },
        save_path,
    )
    print(f"Merged checkpoint saved to: {save_path}")