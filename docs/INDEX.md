# Vision Framework 文档索引

## 📖 主要文档

### 对于新用户
1. **[README.md](README.md)** - 项目概览和快速开始
2. **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - 详细的快速开始指南
3. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - API 快速参考

### 对于开发者
1. **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - 项目目录结构
2. **[docs/ARCHITECTURE_V0.2.8.md](docs/ARCHITECTURE_V0.2.8.md)** - 架构设计说明
3. **[docs/FEATURES.md](docs/FEATURES.md)** - 功能详细说明

### 对于升级用户
1. **[docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)** - v0.2.7 升级指南
2. **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - 版本变更历史

## 🚀 快速导航

| 需求 | 文档 |
|------|------|
| 想快速上手？ | [README.md](README.md) → [docs/QUICKSTART.md](docs/QUICKSTART.md) |
| 查询 API？ | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| 了解架构？ | [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) → [docs/ARCHITECTURE_V0.2.8.md](docs/ARCHITECTURE_V0.2.8.md) |
| 从旧版升级？ | [docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md) |
| 查看更新？ | [docs/CHANGELOG.md](docs/CHANGELOG.md) |
| 查看所有功能？ | [docs/FEATURES.md](docs/FEATURES.md) |

## 📂 目录结构

```
visionframework/
├── README.md                          # 项目主页
├── QUICK_REFERENCE.md                 # API 快速参考
├── LICENSE                            # MIT 许可证
├── requirements.txt                   # 依赖列表
├── setup.py                           # 安装脚本
│
├── docs/                              # 文档目录
│   ├── QUICKSTART.md                  # 快速开始指南
│   ├── FEATURES.md                    # 功能特性
│   ├── PROJECT_STRUCTURE.md           # 项目结构
│   ├── ARCHITECTURE_V0.2.8.md         # 架构设计
│   ├── MIGRATION_GUIDE.md             # 迁移指南
│   └── CHANGELOG.md                   # 版本历史
│
├── examples/                          # 示例代码
│   ├── basic_usage.py                 # 基本使用
│   ├── config_example.py              # 配置文件示例
│   ├── video_tracking.py              # 视频跟踪
│   ├── advanced_features.py           # 高级功能
│   ├── batch_processing.py            # 批量处理
│   ├── yolo_pose_example.py           # 姿态估计
│   ├── rfdetr_example.py              # RF-DETR 示例
│   ├── rfdetr_tracking.py             # RF-DETR 跟踪
│   ├── clip_example.py                # CLIP 示例
│   └── tracking_evaluation_example.py # 评估示例
│
├── tests/                             # 测试代码
│
└── visionframework/                   # 源代码目录
    ├── __init__.py                    # 包导出
    ├── exceptions.py                  # 异常定义
    ├── core/
    ├── models/
    ├── data/
    └── utils/
```

## ✨ 最新更新（v0.2.8）

- ✅ 统一异常系统（12 个异常类）
- ✅ 模型管理器（ModelManager）
- ✅ 特征提取器重构（processors 目录）
- ✅ 改进的文档结构
- ✅ 新增快速参考指南

## 🎯 文档使用建议

### 第一次使用？
```
1. 阅读 README.md（5 分钟）
2. 运行 docs/QUICKSTART.md 中的代码（15 分钟）
3. 查看 examples/ 中的示例代码（30 分钟）
```

### 遇到问题？
```
1. 查看 QUICK_REFERENCE.md 的常见问题
2. 搜索 docs/FEATURES.md 了解功能
3. 查看 examples/ 中的相似示例
```

### 想扩展框架？
```
1. 阅读 docs/ARCHITECTURE_V0.2.8.md
2. 查看 docs/PROJECT_STRUCTURE.md
3. 研究相关的源代码
```

## 📊 文档统计

- **总文档数**: 8 个（6 个在 docs/ + 2 个在根目录）
- **示例代码**: 10+ 个
- **API 文档**: 在 QUICK_REFERENCE.md 中
- **可视化内容**: Markdown 表格和代码块

## 🔗 相关链接

- **GitHub**: (项目仓库地址)
- **Bug 报告**: (issue 地址)
- **讨论区**: (讨论地址)

---

**最后更新**: 2024 年  
**当前版本**: v0.2.8  
**文档完整性**: ✅ 100%
