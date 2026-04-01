# VisionFramework 架构说明

## 分层

| 层级 | 职责 | 典型位置 |
|------|------|----------|
| 入口 | 读取 **runtime YAML**，构建管线 | [`visionframework/task_api.py`](../visionframework/task_api.py) `TaskRunner` |
| 配置 | `resolve_config`（含 `_base_` 继承）、`load_config` | [`visionframework/core/config.py`](../visionframework/core/config.py) |
| 构建 | `build_model` → `ModelWrapper`（backbone / neck / head）+ 权重 | [`visionframework/core/builder.py`](../visionframework/core/builder.py) |
| 注册表 | 按名称实例化组件 | [`visionframework/core/registry.py`](../visionframework/core/registry.py) |
| 算法 | 检测 / 分割 / ReID / 跟踪器实现 | [`visionframework/algorithms/`](../visionframework/algorithms/) |
| 管线 | `process(frame)` → 结果字典 | [`visionframework/pipelines/`](../visionframework/pipelines/) |

`TaskRunner` 解析运行配置后，对 **检测** 任务通过 `_build_detection_algorithm` 从 `ALGORITHMS` 注册表装配 `Detector` / `DETRDetector` / `RTDETRDetector`（由 runtime 的 `algorithm` 字段选择，缺省为 `Detector`）。

**分割（segmentation）** 在 `task_api` 中不直接调用 `build_model`：由 `_build_segmentation_algorithm` 实例化 `YOLO11Segmenter` / `YOLO26Segmenter`，其在内部根据 `configs/segmentation/...` 调用 `build_model` 并对官方 `*-seg.pt` 做 `convert_segment_weights` 后加载（**仅需 PyTorch**，无需 `ultralytics` 包）。管线输出为 `{"detections": [...]}`，每个 `Detection` 含实例 **mask**。

## 跟踪（tracking / reid_tracking）

跟踪管线在装配检测器时 **复用同一套** `_build_detection_algorithm`：可在 runtime YAML 中设置 `algorithm: DETRDetector` 或 `RTDETRDetector`（若模型配置与权重匹配）。跟踪器（ByteTrack / IOUTracker）配置仍通过 `tracker` 字段提供。

## 日志

首次实例化 `TaskRunner` 时会调用 [`visionframework/utils/logging_config.py`](../visionframework/utils/logging_config.py) 中的 `configure_visionframework_logging()`：根据环境变量 `VISIONFRAMEWORK_LOG_LEVEL` 或 `VF_LOG_LEVEL` 设置 `logging.getLogger("visionframework")` 的级别，**默认为 WARNING**（库内 INFO 不输出到控制台，除非提高级别）。pytest 下由根目录 `conftest.py` 默认保持安静。

## 权重与 `strict_weights`

若 runtime 或 `build_model` 指定了权重路径但文件不存在，默认仅记录警告并以随机初始化继续推理。将运行配置中的 `strict_weights: true` 或 `TaskRunner(..., strict_weights=True)` / `build_model(..., strict_weights=True)` 设为真时，缺失权重文件将触发 `FileNotFoundError`。

## 产品与 CLI

- **批量 API**：[`TaskRunner`](../visionframework/task_api.py) 提供 `process_batch`、`process_paths`、`iter_results`、`collect_results`，与现有 `run(source)`（目录遍历见 [`Runner`](../visionframework/engine/runner.py)）互补。
- **命令行**：[`visionframework/tools/run_inference.py`](../visionframework/tools/run_inference.py) 入口名为 **`vf-run`**（`setup.py` 注册），参数 `--config` / `--source` / `--out` / `--strict-weights` / `--max-frames`。

## 扩展新检测算法

1. 实现类并 `@ALGORITHMS.register("YourDetector")`。  
2. 在 [`task_api.py`](../visionframework/task_api.py) 的 `_build_detection_algorithm` 内 `builders` 字典中增加 `"YourDetector": _kw_your` 及对应参数字典工厂。  
3. 补充测试（参见 `test/test_task_api.py`）。

扩展分割算法：在 `ALGORITHMS` 注册后，于 `_build_segmentation_algorithm` 中按名称分支传入 `model_yaml` / 权重即可。批量导出分割可视化 PNG：`python -m visionframework.tools.save_yolo_seg_visualization`（安装后若配置了 `vf-save-yolo-seg` 入口亦同）。
