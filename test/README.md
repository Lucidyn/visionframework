# 测试说明

## 运行

在**仓库根目录**执行：

```bash
pip install -e ".[dev]"
pytest
```

默认选项在 `pyproject.toml` 的 `[tool.pytest.ini_options]` 中配置（如 `addopts = "-q --tb=short"`）。

## 目录结构（摘要）

| 路径 | 内容 |
|------|------|
| `test/core/` | 配置加载、注册表、`build_model` 等 |
| `test/models/` | Backbone / neck / head / RT-DETR 模块等 |
| `test/algorithms/` | 检测器、跟踪器；含 RT-DETR 预训练相关用例 |
| `test/utils/` | bbox、NMS、过滤、可视化等 |
| `test/pipelines/` | 管线构建与行为 |
| `test/fixtures/` | 可选静态资源（如 `bus.jpg`，供官方权重测试） |

## Pytest 标记

| 标记 | 含义 |
|------|------|
| `rtdetr_official` | 需要 **`RTDETR_L_PT`** 与 **`RTDETR_X_PT`** 指向 Ultralytics 官方 COCO 权重 `rtdetr-l.pt`、`rtdetr-x.pt`。未设置或文件不存在时，对应用例 **skip**。 |

运行仅含该标记的用例：

```bash
pytest -m rtdetr_official
```

## RT-DETR 相关

- **`test/algorithms/test_rtdetr_pretrained.py`**
  - `test_rtdetr_taskrunner_image_smoke`（参数化 l/x）：随机初始化权重写入临时文件，验证 `TaskRunner` + 图像推理链路，**不下载权重**。
  - `test_rtdetr_official_pt_bus_detections`（`@pytest.mark.rtdetr_official`）：用官方 `.pt` 经 `convert_ultralytics_rtdetr_hg` 转换后，在 `test/fixtures/bus.jpg`（若存在）上断言至少检出 1 个目标。
- **`test/models/test_rtdetr.py`**：模型配置与构建等单元测试。

官方 `.pt` 许可见根目录 **`NOTICE`**。

## 可选：与 Ultralytics 对齐的验证依赖

安装 `pip install -e ".[rtdetr-verify]"` 后，可使用与 Ultralytics 逐张量/推理对齐的额外测试（若仓库中提供；详见各测试文件 docstring）。
