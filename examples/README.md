# 示例

所有示例都使用统一的 `Vision` API。

## 基础示例

| 文件 | 功能 | 核心用法 |
|------|------|---------|
| `00_basic_detection.py` | 基本目标检测 | `Vision()` + `v.run()` |
| `01_detection_with_tracking.py` | 检测 + 跟踪 | `Vision(track=True)` |
| `02_simplified_api.py` | 从配置文件加载 | `Vision.from_config()` |
| `03_pose_estimation.py` | 姿态估计 | `Vision(pose=True)` |
| `04_segmentation.py` | 实例分割 | `Vision(segment=True)` |
| `05_video_processing.py` | 视频 / 摄像头处理 | `v.run()` + `v.draw()` |

## 进阶示例

| 文件 | 功能 |
|------|------|
| `08_model_tools_example.py` | 模型工具（量化、剪枝、增强等） |
| `09_multimodal_processing.py` | 多模态处理 |
| `10_batch_processing.py` | 批处理 |
| `11_custom_component.py` | 自定义组件 |
| `12_result_export.py` | 结果导出 |

## 快速运行

```bash
# 安装
pip install -e .

# 运行
python examples/basic/00_basic_detection.py
```

## API 速查

```python
from visionframework import Vision

# 方式一：直接创建
v = Vision(model="yolov8n.pt", track=True, pose=True)

# 方式二：从配置文件创建
v = Vision.from_config("config.json")

# 处理任意来源
for frame, meta, result in v.run(source):
    print(result["detections"])
    print(result["tracks"])
    print(result["poses"])
```

`source` 支持：图片路径、视频路径、摄像头 (0)、RTSP 流、文件夹、路径列表、numpy 数组。
