Examples (recreated)

This `examples/` folder contains simple, focused example scripts. Each file
demonstrates one feature and is intentionally minimal so new users can run
and understand them quickly.

Files:
- `detect_basic.py` — 单图像检测与结果可视化（输出 `output_detect_basic.jpg`）
- `pipeline_simple.py` — 使用 `VisionPipeline` 的最小示例（输出 `output_pipeline_simple.jpg`）
- `config_example.yaml` — 最小配置示例，用于 `Config` 或示例脚本
- `ocr_example.py` — 使用 `pytesseract` 的简单 OCR 示范（可选依赖）

运行示例：在项目根目录执行，例如：

```bash
python examples/detect_basic.py path/to/image.jpg
python examples/pipeline_simple.py
python examples/ocr_example.py
```

如果你希望把原先的进阶示例保留下来作为 `examples/advanced/`，我可以把已删除的文件恢复到该目录。要我恢复吗？
