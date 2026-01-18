# 项目结构

仓库主要目录说明：

- `visionframework/`：框架核心代码
  - `core/`：主流程组件（Detector、Tracker、Pipeline、Processor）
  - `data/`：数据类（Detection、Track、Pose）
  - `models/`：模型管理相关代码
  - `utils/`：工具函数（配置、可视化、IO）
- `examples/`：按功能组织的示例脚本
- `docs/`：本地文档（快速开始、API 速查、架构等）
- `tests/`：单元/集成测试
- `requirements.txt` / `pyproject.toml`：依赖和开发配置

如何扩展：
- 按能力编写模块（例如新 Detector），实现框架定义的接口并注册到 `core` 中。
- 使用 `examples/` 提供的最小示例作为集成参考。
