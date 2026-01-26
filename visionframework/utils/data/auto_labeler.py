#!/usr/bin/env python3
"""
自动标注工具 - 基于Vision Framework的自动标注系统

该模块提供自动标注功能，支持单张图像和视频的自动检测、跟踪和标注生成，
并支持多种标注格式导出，便于模型训练和数据标注。

Example:
    ```python
    from visionframework.utils import AutoLabeler
    from visionframework import Config
    
    # 初始化自动标注器
    auto_labeler = AutoLabeler({
        "detector_config": Config.get_default_detector_config(),
        "enable_tracking": True,
        "output_format": "coco",
        "output_path": "output/annotations"
    })
    
    # 标注单张图像
    auto_labeler.label_image("image.jpg")
    
    # 标注视频
    auto_labeler.label_video("video.mp4")
    
    # 批量标注图像
    auto_labeler.label_batch(["image1.jpg", "image2.jpg"])
    ```
"""

from typing import List, Dict, Any, Optional, Tuple
import os
from pathlib import Path
import cv2
import numpy as np

from visionframework.core.detector import Detector
from visionframework.core.tracker import Tracker
from visionframework.core.pipeline import VisionPipeline
from visionframework.data.detection import Detection
from visionframework.data.track import Track
from .export import ResultExporter
from ..io.video_utils import process_video
from ..monitoring.logger import get_logger

logger = get_logger(__name__)


class AutoLabeler:
    """
    自动标注器类，用于自动生成图像和视频的标注
    
    该类整合了检测器、跟踪器和结果导出功能，提供统一的自动标注接口，
    支持单张图像、视频和批量图像的自动标注，并支持多种标注格式导出。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化自动标注器
        
        Args:
            config: 配置字典，包含以下可选键：
                - detector_config: 检测器配置，默认使用YOLOv8n模型
                - tracker_config: 跟踪器配置，默认使用IOU跟踪器
                - enable_tracking: 是否启用跟踪（仅适用于视频），默认False
                - output_format: 输出格式，可选值："json", "csv", "coco"，默认"coco"
                - output_path: 输出路径，默认"output/annotations"
                - conf_threshold: 默认置信度阈值，默认0.25
                - category_thresholds: 每个类别的置信度阈值，默认None
                - class_names: 类别名称列表，默认使用检测器自带类别
        """
        self.config = config or {}
        self.detector: Optional[Detector] = None
        self.tracker: Optional[Tracker] = None
        self.pipeline: Optional[VisionPipeline] = None
        self.exporter: Optional[ResultExporter] = None
        
        # 配置参数
        self.enable_tracking: bool = self.config.get("enable_tracking", False)
        self.output_format: str = self.config.get("output_format", "coco")
        self.output_path: str = self.config.get("output_path", "output/annotations")
        self.conf_threshold: float = self.config.get("conf_threshold", 0.25)
        self.category_thresholds: Optional[Dict[str, float]] = self.config.get("category_thresholds")
        
        # 确保输出目录存在
        os.makedirs(self.output_path, exist_ok=True)
        
        # 初始化组件
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """初始化自动标注器的组件"""
        detector_config = self.config.get("detector_config", {})
        
        # 应用类别特定的置信度阈值
        if self.category_thresholds:
            detector_config["category_thresholds"] = self.category_thresholds
        
        # 初始化检测器
        self.detector = Detector(detector_config)
        
        if not self.detector.initialize():
            raise RuntimeError("Failed to initialize detector for auto labeling")
        
        # 初始化跟踪器（如果启用）
        if self.enable_tracking:
            tracker_config = self.config.get("tracker_config", {})
            self.tracker = Tracker(tracker_config)
            
            if not self.tracker.initialize():
                logger.warning("Failed to initialize tracker, disabling tracking")
                self.enable_tracking = False
                self.tracker = None
            else:
                # 初始化视觉管道
                pipeline_config = {
                    "detector_config": detector_config,
                    "tracker_config": tracker_config,
                    "enable_tracking": True
                }
                self.pipeline = VisionPipeline(pipeline_config)
                if not self.pipeline.initialize():
                    logger.warning("Failed to initialize pipeline, using separate detector and tracker")
                    self.pipeline = None
        
        # 初始化结果导出器
        self.exporter = ResultExporter()
    
    def label_image(self, image_path: str, output_name: Optional[str] = None) -> Dict[str, Any]:
        """
        标注单张图像
        
        Args:
            image_path: 图像路径
            output_name: 输出文件名，默认使用图像文件名
            
        Returns:
            Dict[str, Any]: 标注结果
        """
        logger.info(f"开始标注图像: {image_path}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 执行检测
        detections = self.detector.detect(image)
        
        # 过滤检测结果（根据类别特定阈值）
        if self.category_thresholds:
            detections = self._filter_detections_by_category_threshold(detections)
        
        # 生成输出文件名
        if output_name is None:
            output_name = Path(image_path).stem
        
        # 导出标注
        output_file = os.path.join(self.output_path, f"{output_name}.{self.output_format}")
        
        # 根据格式导出
        if self.output_format == "coco":
            # COCO格式导出需要图像信息
            image = cv2.imread(image_path)
            if image is not None:
                h, w = image.shape[:2]
                image_info = {
                    "width": w,
                    "height": h,
                    "file_name": os.path.basename(image_path)
                }
                self.exporter.export_to_coco_format(
                    detections,
                    0,  # 简单起见，使用0作为图像ID
                    image_info,
                    output_file
                )
            else:
                logger.error(f"无法读取图像 {image_path}，无法导出COCO格式")
        elif self.output_format == "json":
            # 导出为JSON格式
            self.exporter.export_detections_to_json(
                detections,
                output_file,
                metadata={"image_path": image_path}
            )
        elif self.output_format == "csv":
            # 导出为CSV格式
            self.exporter.export_detections_to_csv(
                detections,
                output_file
            )
        else:
            logger.error(f"不支持的输出格式: {self.output_format}")
        
        logger.info(f"图像标注完成，结果保存到: {output_file}")
        return {
            "image_path": image_path,
            "detections": detections,
            "output_file": output_file,
            "num_detections": len(detections)
        }
    
    def label_video(self, video_path: str, output_name: Optional[str] = None) -> Dict[str, Any]:
        """
        标注视频
        
        Args:
            video_path: 视频路径
            output_name: 输出文件名，默认使用视频文件名
            
        Returns:
            Dict[str, Any]: 标注结果
        """
        logger.info(f"开始标注视频: {video_path}")
        
        # 生成输出文件名
        if output_name is None:
            output_name = Path(video_path).stem
        
        # 初始化结果列表
        all_detections: List[List[Detection]] = []
        all_tracks: List[List[Track]] = []
        frame_count = 0
        
        # 视频处理回调函数
        def process_frame(frame: np.ndarray, frame_idx: int) -> bool:
            nonlocal all_detections, all_tracks, frame_count
            
            if self.enable_tracking and self.pipeline:
                # 使用管道进行检测和跟踪
                tracks = self.pipeline.process(frame)
                all_tracks.append(tracks)
            elif self.enable_tracking and self.tracker:
                # 先检测，再跟踪
                detections = self.detector.detect(frame)
                if self.category_thresholds:
                    detections = self._filter_detections_by_category_threshold(detections)
                tracks = self.tracker.process(detections, frame)
                all_tracks.append(tracks)
                all_detections.append(detections)
            else:
                # 仅检测
                detections = self.detector.detect(frame)
                if self.category_thresholds:
                    detections = self._filter_detections_by_category_threshold(detections)
                all_detections.append(detections)
            
            frame_count += 1
            return True
        
        # 处理视频
        process_video(video_path, process_frame)
        
        # 导出标注
        output_file = os.path.join(self.output_path, f"{output_name}.{self.output_format}")
        
        # 根据格式导出
        if self.output_format == "coco":
            # COCO格式不直接支持视频，导出第一个帧的检测结果
            if all_detections:
                # 读取视频的第一帧获取尺寸信息
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        h, w = frame.shape[:2]
                        image_info = {
                            "width": w,
                            "height": h,
                            "file_name": f"frame_0.jpg"
                        }
                        self.exporter.export_to_coco_format(
                            all_detections[0],
                            0,  # 简单起见，使用0作为图像ID
                            image_info,
                            output_file
                        )
                    cap.release()
            else:
                logger.error("没有检测结果，无法导出COCO格式")
        elif self.output_format == "json":
            # 导出为JSON格式
            if self.enable_tracking and all_tracks:
                # 由于ResultExporter没有直接导出批量跟踪结果的方法，我们将每个帧的结果单独处理
                video_results = []
                for frame_idx, tracks in enumerate(all_tracks):
                    video_results.append({
                        "frame_idx": frame_idx,
                        "tracks": [track.to_dict() for track in tracks]
                    })
                self.exporter.export_video_results_to_json(
                    video_results,
                    output_file,
                    video_info={"video_path": video_path, "total_frames": frame_count}
                )
            else:
                # 同样，处理批量检测结果
                video_results = []
                for frame_idx, detections in enumerate(all_detections):
                    video_results.append({
                        "frame_idx": frame_idx,
                        "detections": [det.to_dict() for det in detections]
                    })
                self.exporter.export_video_results_to_json(
                    video_results,
                    output_file,
                    video_info={"video_path": video_path, "total_frames": frame_count}
                )
        else:
            logger.error(f"不支持的输出格式: {self.output_format}")
        
        logger.info(f"视频标注完成，处理了 {frame_count} 帧，结果保存到: {output_file}")
        return {
            "video_path": video_path,
            "frame_count": frame_count,
            "output_file": output_file,
            "num_detections": sum(len(dets) for dets in all_detections),
            "num_tracks": sum(len(tracks) for tracks in all_tracks) if all_tracks else 0
        }
    
    def label_batch(self, image_paths: List[str], batch_size: int = 4) -> List[Dict[str, Any]]:
        """
        批量标注图像
        
        Args:
            image_paths: 图像路径列表
            batch_size: 批处理大小
            
        Returns:
            List[Dict[str, Any]]: 标注结果列表
        """
        logger.info(f"开始批量标注，共 {len(image_paths)} 张图像")
        
        results = []
        
        # 批量处理图像
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # 读取批量图像
            batch_images = []
            valid_paths = []
            
            for img_path in batch_paths:
                img = cv2.imread(img_path)
                if img is not None:
                    batch_images.append(img)
                    valid_paths.append(img_path)
                else:
                    logger.warning(f"无法读取图像: {img_path}")
            
            if not batch_images:
                continue
            
            # 执行批量检测
            batch_detections = self.detector.detect_batch(batch_images)
            
            # 处理每个图像的检测结果
            for j, (img_path, detections) in enumerate(zip(valid_paths, batch_detections)):
                # 过滤检测结果
                if self.category_thresholds:
                    detections = self._filter_detections_by_category_threshold(detections)
                
                # 导出标注
                output_name = Path(img_path).stem
                output_file = os.path.join(self.output_path, f"{output_name}.{self.output_format}")
                
                if self.output_format == "coco":
                    # COCO格式需要单独处理
                    image = cv2.imread(img_path)
                    if image is not None:
                        h, w = image.shape[:2]
                        image_info = {
                            "width": w,
                            "height": h,
                            "file_name": os.path.basename(img_path)
                        }
                        self.exporter.export_to_coco_format(
                            detections,
                            i * batch_size + j,  # 使用唯一的图像ID
                            image_info,
                            output_file
                        )
                    else:
                        logger.error(f"无法读取图像 {img_path}，无法导出COCO格式")
                elif self.output_format == "json":
                    # 导出为JSON格式
                    self.exporter.export_detections_to_json(
                        detections,
                        output_file,
                        metadata={"image_path": img_path}
                    )
                elif self.output_format == "csv":
                    # 导出为CSV格式
                    self.exporter.export_detections_to_csv(
                        detections,
                        output_file
                    )
                else:
                    logger.error(f"不支持的输出格式: {self.output_format}")
                
                results.append({
                    "image_path": img_path,
                    "detections": detections,
                    "output_file": output_file,
                    "num_detections": len(detections)
                })
        
        logger.info(f"批量标注完成，共处理 {len(results)} 张图像")
        return results
    
    def _filter_detections_by_category_threshold(self, detections: List[Detection]) -> List[Detection]:
        """
        根据类别特定的置信度阈值过滤检测结果
        
        Args:
            detections: 检测结果列表
            
        Returns:
            List[Detection]: 过滤后的检测结果
        """
        filtered_detections = []
        
        for detection in detections:
            # 获取该类别的置信度阈值
            cat_threshold = self.category_thresholds.get(
                detection.class_name,
                self.conf_threshold
            )
            
            if detection.confidence >= cat_threshold:
                filtered_detections.append(detection)
        
        return filtered_detections
    
    def cleanup(self) -> None:
        """清理资源"""
        if self.detector:
            self.detector.cleanup()
        
        if self.tracker:
            self.tracker.cleanup()
        
        if self.pipeline:
            self.pipeline.cleanup()
    
    def __del__(self) -> None:
        """析构函数，确保资源释放"""
        self.cleanup()
