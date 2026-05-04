#!/usr/bin/env python3
# Copyright (c) Facebook, Inc.

"""Run a video action recognition demo with per-clip JSONL logging."""

import datetime
import json
import os
import time
from typing import Iterable

import numpy as np
import torch
import tqdm

from slowfast.datasets import loader
from slowfast.utils import logging
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.ava_demo_precomputed_boxes import (
    AVAVisualizerWithPrecomputedBox,
)
from slowfast.visualization.demo_loader import ThreadVideoManager, VideoManager
from slowfast.visualization.predictor import ActionPredictor
from slowfast.visualization.video_visualizer import VideoVisualizer

logger = logging.get_logger(__name__)


def _dump_clip_json(
    *,
    jsonl_path: str,
    task,
    preds_tensor: torch.Tensor,
    class_names: Iterable[str],
    num_frames_clip: int,
    cfg,
) -> None:
    """Append one record to metrics.jsonl for the current clip."""
    clip_ms = float(getattr(task, "infer_ms", 0.0))
    clips_per_sec = (1000.0 / clip_ms) if clip_ms > 0 else 0.0
    ms_per_frame = clip_ms / float(num_frames_clip) if num_frames_clip > 0 else 0.0
    fps_model = (1000.0 / ms_per_frame) if ms_per_frame > 0 else 0.0

    probs = torch.softmax(preds_tensor, dim=-1)
    k_cfg = getattr(getattr(cfg, "TENSORBOARD").MODEL_VIS, "TOPK_PREDS", 5)
    k = max(1, int(k_cfg))
    if probs.ndim == 1:
        probs = probs.unsqueeze(0)

    topk_vals, topk_idx = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
    vals = topk_vals[0].detach().cpu().tolist()
    idxs = topk_idx[0].detach().cpu().tolist()

    class_names = list(class_names) if class_names is not None else []
    topk_list = []
    for score, cls_idx in zip(vals, idxs):
        name = class_names[cls_idx] if 0 <= cls_idx < len(class_names) else f"class_{cls_idx}"
        topk_list.append({"cls_id": int(cls_idx), "cls_name": name, "score": float(score)})

    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "clip_id": int(task.id),
        "num_frames_clip": int(num_frames_clip),
        "num_buffer_frames": int(getattr(task, "num_buffer_frames", 0)),
        "timing_ms_clip": round(clip_ms, 3),
        "clips_per_sec": round(clips_per_sec, 6),
        "ms_per_frame": round(ms_per_frame, 3),
        "fps_model": round(fps_model, 6),
        "action_preds": probs.detach().cpu().numpy().tolist(),
        "topk": topk_list,
    }

    with open(jsonl_path, "a") as handle:
        handle.write(json.dumps(record) + "\n")


def _print_clip_pretty(
    *,
    task,
    preds_tensor: torch.Tensor,
    class_names: Iterable[str],
    num_frames_clip: int,
) -> None:
    """Print predictions clip by clip."""
    pt = preds_tensor.detach().cpu()
    if pt.ndim == 2 and pt.size(0) == 1:
        pt = pt.squeeze(0)
    elif pt.ndim != 1:
        pt = pt[0]

    probs = torch.softmax(pt, dim=-1)
    top1_val, top1_idx = torch.max(probs, dim=-1)
    top1_idx = int(top1_idx.item())
    top1_val = float(top1_val.item())

    class_names = list(class_names) if class_names is not None else []
    top1_name = class_names[top1_idx] if 0 <= top1_idx < len(class_names) else f"class_{top1_idx}"

    all_pairs = []
    for cls_id, score in enumerate(probs.tolist()):
        name = class_names[cls_id] if 0 <= cls_id < len(class_names) else f"class_{cls_id}"
        all_pairs.append(f"{name}: {score:.4f}")
    all_line = " | ".join(all_pairs)

    clip_ms = float(getattr(task, "infer_ms", 0.0))
    clips_per_sec = (1000.0 / clip_ms) if clip_ms > 0 else 0.0
    ms_per_frame = clip_ms / float(num_frames_clip) if num_frames_clip > 0 else 0.0
    fps_model = (1000.0 / ms_per_frame) if ms_per_frame > 0 else 0.0

    print(f"[CLIP {int(task.id):05d}]")
    print(f"    --> predictions (TOP1):  {top1_name}   {top1_val:.4f}")
    print(f"    --> predictions (ALL):   {all_line}")
    print(
        f"    --> tempo: forward: {clip_ms:.2f} ms  "
        f"(~{clips_per_sec:.3f} clips/s, ~{fps_model:.2f} FPS-model, ~{ms_per_frame:.4f} ms/frame)"
    )


def run_demo(cfg, frame_provider) -> Iterable:
    """Run demo visualization and write metrics JSONL for each processed clip."""
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    logging.setup_logging(cfg.OUTPUT_DIR)
    logger.info("Run demo with config:")
    logger.info(cfg)

    common_classes = (
        cfg.DEMO.COMMON_CLASS_NAMES if len(cfg.DEMO.LABEL_FILE_PATH) != 0 else None
    )

    video_vis = VideoVisualizer(
        num_classes=cfg.MODEL.NUM_CLASSES,
        class_names_path=cfg.DEMO.LABEL_FILE_PATH,
        top_k=cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
        thres=cfg.DEMO.COMMON_CLASS_THRES,
        lower_thres=cfg.DEMO.UNCOMMON_CLASS_THRES,
        common_class_names=common_classes,
        colormap=cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
        mode=cfg.DEMO.VIS_MODE,
    )

    jsonl_env = os.getenv("METRICS_JSONL")
    if jsonl_env:
        jsonl_path = os.path.abspath(jsonl_env)
    else:
        jsonl_path = os.path.join(os.path.abspath(cfg.OUTPUT_DIR), "metrics.jsonl")
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    logger.info(f"[DEMO] metrics jsonl path: {jsonl_path}")
    try:
        with open(jsonl_path, "a"):
            pass
    except Exception as exc:
        logger.error(f"[DEMO] cannot create metrics file at {jsonl_path}: {exc}")

    async_vis = AsyncVis(video_vis, n_workers=getattr(cfg.DEMO, "NUM_VIS_INSTANCES", 1))
    if cfg.NUM_GPUS <= 1:
        model = ActionPredictor(cfg=cfg, async_vis=async_vis)
    else:
        model = AsyncDemo(cfg=cfg, async_vis=async_vis)

    seq_len = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
    assert cfg.DEMO.BUFFER_SIZE <= seq_len // 2, (
        "BUFFER_SIZE cannot be greater than half of sequence length."
    )

    records_written = 0

    if cfg.DEMO.AS_TEST:
        demo_loader = loader.construct_loader(cfg, "test")
        model.predictor.perform_test(demo_loader)

    if not cfg.DEMO.AS_TEST:
        num_task = 0
        frame_provider.start()
        for able_to_read, task in frame_provider:
            if not able_to_read:
                break
            if task is None:
                time.sleep(0.02)
                continue
            num_task += 1
            model.put(task)
            try:
                task = model.get()
                _print_clip_pretty(
                    task=task,
                    preds_tensor=task.action_preds,
                    class_names=video_vis.class_names,
                    num_frames_clip=cfg.DATA.NUM_FRAMES,
                )
                try:
                    _dump_clip_json(
                        jsonl_path=jsonl_path,
                        task=task,
                        preds_tensor=task.action_preds.detach().cpu(),
                        class_names=video_vis.class_names,
                        num_frames_clip=cfg.DATA.NUM_FRAMES,
                        cfg=cfg,
                    )
                    records_written += 1
                except Exception as exc:
                    logger.error(f"[DEMO] failed to write metrics jsonl: {exc}")
                num_task -= 1
                yield task
            except IndexError:
                continue

        while num_task != 0:
            try:
                task = model.get()
                _print_clip_pretty(
                    task=task,
                    preds_tensor=task.action_preds,
                    class_names=video_vis.class_names,
                    num_frames_clip=cfg.DATA.NUM_FRAMES,
                )
                try:
                    _dump_clip_json(
                        jsonl_path=jsonl_path,
                        task=task,
                        preds_tensor=task.action_preds.detach().cpu(),
                        class_names=video_vis.class_names,
                        num_frames_clip=cfg.DATA.NUM_FRAMES,
                        cfg=cfg,
                    )
                    records_written += 1
                except Exception as exc:
                    logger.error(f"[DEMO] failed to write metrics jsonl: {exc}")
                num_task -= 1
                yield task
            except IndexError:
                continue

    if records_written == 0:
        logger.warning("[DEMO] no clips processed or no records written to metrics jsonl.")


def demo(cfg):
    """Run inference on an input video or webcam stream."""
    print(
        f"[DEMO] NUM_FRAMES={cfg.DATA.NUM_FRAMES}, "
        f"SAMPLING_RATE={cfg.DATA.SAMPLING_RATE}, "
        f"SEQ_LEN={cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE}, "
        f"BUFFER_SIZE={cfg.DEMO.BUFFER_SIZE}, "
        f"THREADS={cfg.DEMO.THREAD_ENABLE}, "
        f"NUM_CLIPS_SKIP={getattr(cfg.DEMO, 'NUM_CLIPS_SKIP', 0)}"
    )
    logger.info(
        f"[DEMO] expecting clip length (frames): {cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE}"
    )

    if cfg.ONLY_PRED:
        logger.info("Esecuzione demo (solo predizione, senza visualizzazione)...")
        return

    if cfg.DETECTION.ENABLE and cfg.DEMO.PREDS_BOXES != "":
        precomputed_box_vis = AVAVisualizerWithPrecomputedBox(cfg)
        precomputed_box_vis()
        return

    start = time.time()
    frame_provider = ThreadVideoManager(cfg) if cfg.DEMO.THREAD_ENABLE else VideoManager(cfg)

    for task in tqdm.tqdm(run_demo(cfg, frame_provider)):
        frame_provider.display(task)

    frame_provider.join()
    frame_provider.clean()
    logger.info("Finish demo in: {:.3f}s".format(time.time() - start))