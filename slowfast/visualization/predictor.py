#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import queue
import cv2
import torch

import slowfast.utils.checkpoint as cu
from slowfast.models import build_model
from slowfast.utils import logging
from slowfast.visualization.utils import process_cv2_inputs
import torchvision.transforms.functional as F
import os
import time

logger = logging.get_logger(__name__)

def _resolve_checkpoint_state(checkpoint, cfg, scope):
    checkpoint_model = checkpoint["model_state"]
    is_delta = bool(checkpoint.get("is_delta", False))
    use_merge = bool(getattr(scope, "MERGE_WITH_BASE", False) or is_delta)
    if not use_merge:
        return checkpoint_model

    base_file = getattr(scope, "MERGE_BASE_FILE", None) or checkpoint.get("base_checkpoint")
    assert base_file is not None, "Missing base checkpoint for merged delta reconstruction."
    assert os.path.exists(base_file), "Base checkpoint '{}' not found".format(base_file)

    logger.info("Loading base checkpoint from {} for delta reconstruction.".format(base_file))
    base_ckpt = torch.load(base_file, map_location="cpu")
    base_state = base_ckpt.get("model_state") or base_ckpt.get("state_dict") or base_ckpt
    alpha = float(getattr(scope, "MERGE_ALPHA", 1.0))
    logger.info("Reconstructing merged model with alpha={}.".format(alpha))

    return {
        k: base_state[k].float() + alpha * checkpoint_model.get(k, torch.zeros_like(base_state[k])).float()
        for k in base_state
    }

class Predictor:
    """
    Action Predictor for action recognition.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """
        if cfg.NUM_GPUS:
            self.gpu_id = (
                torch.cuda.current_device() if gpu_id is None else gpu_id
            )

        # Build the video model and print model statistics.
        self.model = build_model(cfg, gpu_id=gpu_id)
        
        if cfg.DEMO.CUSTOM_LOAD:
            custom_load_file = cfg.DEMO.CUSTOM_LOAD_FILE
            logger.info(f"Custom load weights from: {custom_load_file}")
            
            checkpoint = torch.load(custom_load_file, map_location="cpu")
            checkpoint_model = _resolve_checkpoint_state(checkpoint, cfg, cfg.DEMO)
            state_dict = self.model.state_dict()
            
            for key in checkpoint_model.keys():
                if key not in state_dict.keys():
                    logger.info("missing key {} in model.".format(key))
                    
            self.model.load_state_dict(checkpoint_model, strict=False)
        
        self.model.eval()
        self.cfg = cfg

        #if cfg.DETECTION.ENABLE:
        #    self.object_detector = Detectron2Predictor(cfg, gpu_id=self.gpu_id)
                
        logger.info("Start loading model weights.")
        cu.load_test_checkpoint(cfg, self.model)
        logger.info("Finish loading model weights")

    def __call__(self, task):
        """
        Returns the prediction results for the current task.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        #if self.cfg.DETECTION.ENABLE:
        #    task = self.object_detector(task)

        frames = task.frames

        if self.cfg.DEMO.INPUT_FORMAT == "BGR":
            frames = [
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames
            ]
        
        inputs = process_cv2_inputs(frames, self.cfg)

        if self.cfg.NUM_GPUS > 0:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(
                        device=torch.device(self.gpu_id), non_blocking=True
                    )
            else:
                inputs = inputs.cuda(
                    device=torch.device(self.gpu_id), non_blocking=True
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            preds = self.model(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_ms = (time.time() - t0) * 1000
        
        setattr(task, "infer_ms", float(infer_ms))
        if self.cfg.NUM_GPUS:
            preds = preds.cpu()

        preds = preds.detach()
        #debug
        #print(f"[DEBUG] pred shape={tuple(p.shape)}, min={p.min().item():.3f}, max={p.max().item():.3f}, mean={p.mean().item():.3f}")

        #print(f"PREDICTIONS di DEMO:{torch.softmax(preds, dim=-1)}")
        task.add_action_preds(preds)

        return task
    
    def perform_test(self, loader):
        self.model.eval()
        
        for cur_iter, (inputs, labels, video_idx, time, meta) in enumerate(loader):
            
            if self.cfg.NUM_GPUS:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                video_idx = video_idx.cuda()
                
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)
            #print(f"INPUTS SHAPE: {inputs.shape()}")
            preds = self.model(inputs)

            if self.cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

class ActionPredictor:
    """
    Synchronous Action Prediction and Visualization pipeline with AsyncVis.
    """

    def __init__(self, cfg, async_vis=None, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            async_vis (AsyncVis object): asynchronous visualizer.
            gpu_id (Optional[int]): GPU id.
        """
        self.predictor = Predictor(cfg=cfg, gpu_id=gpu_id)
        self.async_vis = async_vis

    def put(self, task):
        """
        Make prediction and put the results in `async_vis` task queue.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames, boxes)
        """
        task = self.predictor(task)
        self.async_vis.get_indices_ls.append(task.id)
        self.async_vis.put(task)

    def get(self):
        """
        Get the visualized clips if any.
        """
        try:
            task = self.async_vis.get()
        except (queue.Empty, IndexError):
            raise IndexError("Results are not available yet.")

        return task


class Detectron2Predictor:
    """
    Wrapper around Detectron2 to return the required predicted bounding boxes
    as a ndarray.
    """

    def __init__(self, cfg, gpu_id=None):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
                slowfast/config/defaults.py
            gpu_id (Optional[int]): GPU id.
        """

        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(cfg.DEMO.DETECTRON2_CFG)
        )
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = cfg.DEMO.DETECTRON2_THRESH
        self.cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_WEIGHTS
        self.cfg.INPUT.FORMAT = cfg.DEMO.INPUT_FORMAT
        if cfg.NUM_GPUS and gpu_id is None:
            gpu_id = torch.cuda.current_device()
        self.cfg.MODEL.DEVICE = (
            "cuda:{}".format(gpu_id) if cfg.NUM_GPUS > 0 else "cpu"
        )

        logger.info("Initialized Detectron2 Object Detection Model.")

        self.predictor = DefaultPredictor(self.cfg)

    def __call__(self, task):
        """
        Return bounding boxes predictions as a tensor.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for action prediction. (e.g. frames)
        Returns:
            task (TaskInfo object): the same task info object but filled with
                prediction values (a tensor) and the corresponding boxes for
                action detection task.
        """
        middle_frame = task.frames[len(task.frames) // 2]
        outputs = self.predictor(middle_frame)
        # Get only human instances
        mask = outputs["instances"].pred_classes == 0
        pred_boxes = outputs["instances"].pred_boxes.tensor[mask]
        task.add_bboxes(pred_boxes)

        return task