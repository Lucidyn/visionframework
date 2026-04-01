# 测试说明

## 运行

在**仓库根目录**执行：

```bash
pip install -e ".[dev]"
pytest
```

默认选项在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 中配置（含 `-m "not yolo_seg"`，排除需下载官方 `*-seg.pt` 的慢测）。

根目录 **`conftest.py`** 会 `setdefault("VISIONFRAMEWORK_LOG_LEVEL", "WARNING")`，避免 `TaskRunner` 等在测试中输出 INFO。调试单测时可临时设置 `VISIONFRAMEWORK_LOG_LEVEL=INFO`。

## 目录结构（摘要）

| 路径 | 内容 |
|------|------|
| `test/core/` | 配置加载、注册表、`build_model` 等 |
| `test/models/` | Backbone / neck / head / RT-DETR 模块等 |
| `test/algorithms/` | 检测器、跟踪器、**YOLO 实例分割**（`test_yolo_segmentation.py`）；含 RT-DETR 预训练相关用例 |
| `test/utils/` | bbox、NMS、过滤、可视化、`logging_config` 等 |
| `test/pipelines/` | 管线构建与行为 |
| `test/test_task_api.py` | `TaskRunner`、`_build_detection_algorithm`、`strict_weights`、跟踪管线冒烟 |
| `test/fixtures/` | 可选静态资源（如 `bus.jpg`，供官方权重测试） |

## Pytest 标记

| 标记 | 含义 |
|------|------|
| `rtdetr_official` | 需要 **`RTDETR_L_PT`** 与 **`RTDETR_X_PT`** 指向 Ultralytics 官方 COCO 权重 `rtdetr-l.pt`、`rtdetr-x.pt`。未设置或文件不存在时，对应用例 **skip**。 |
| `yolo_seg` | YOLO11/YOLO26 全尺寸 `*-seg.pt` 推理 + **Visualizer 叠加 + PNG 往返**（**仅需 PyTorch**，权重由测试从 GitHub release 下载）。测试图优先 `test/fixtures/bus.jpg`。默认 `pytest` **不收集**；显式运行：`pytest -m yolo_seg`。若某权重下载损坏会 **skip**。 |

运行仅含该标记的用例：

```bash
pytest -m rtdetr_official
pytest -m yolo_seg test/algorithms/test_yolo_segmentation.py
```

## YOLO 实例分割（`yolo_seg`）

- **文件**：`test/algorithms/test_yolo_segmentation.py`
- **内容**：对 YOLO11 / YOLO26 各 5 档尺寸做推理；**`Visualizer` 叠加**与 **PNG 写回读**；`TaskRunner` + runtime YAML 各一条。
- **测试图**：优先 `test_bus.jpg`，其次 **`test/fixtures/bus.jpg`**（无网络时避免下载失败）。
- **损坏权重**：若下载的 `.pt` 不完整，相关用例 **`pytest.skip`**。
- **导出可视化**：`python -m visionframework.tools.save_yolo_seg_visualization`（见主 **README**）。

## RT-DETR 相关

- **`test/algorithms/test_rtdetr_pretrained.py`**
  - `test_rtdetr_taskrunner_image_smoke`（参数化 l/x）：随机初始化权重写入临时文件，验证 `TaskRunner` + 图像推理链路，**不下载权重**。
  - `test_rtdetr_official_pt_bus_detections`（`@pytest.mark.rtdetr_official`）：用官方 `.pt` 经 `convert_ultralytics_rtdetr_hg` 转换后，在 `test/fixtures/bus.jpg`（若存在）上断言至少检出 1 个目标。
- **`test/models/test_rtdetr.py`**：模型配置与构建等单元测试。

官方 `.pt` 许可见根目录 **`NOTICE`**。

## 可选：与 Ultralytics 对齐的验证依赖

- **`pip install -e ".[rtdetr-verify]"`**：RT-DETR 等与 Ultralytics 逐张量/推理对齐的额外测试（需安装 `ultralytics`，详见各测试文件 docstring）。
