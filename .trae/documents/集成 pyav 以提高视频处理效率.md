# 集成 pyav 以提高视频处理效率

## 问题分析
当前项目使用 OpenCV 进行视频处理，虽然功能完整，但在处理高分辨率视频时可能效率不够理想。pyav 作为基于 FFmpeg 的视频处理库，通常比 OpenCV 提供更高的性能和更丰富的功能。

## 改进方案

### 1. 新增 PyAVVideoProcessor 类
- 创建 `PyAVVideoProcessor` 类，基于 pyav 实现视频读取
- 保持与现有 `VideoProcessor` 相同的接口，确保向后兼容
- 支持视频文件、摄像头和网络流
- 利用 pyav 的硬件加速功能

### 2. 新增 PyAVVideoWriter 类
- 创建 `PyAVVideoWriter` 类，基于 pyav 实现视频写入
- 保持与现有 `VideoWriter` 相同的接口
- 支持更多编解码器和容器格式
- 提供更高的写入效率

### 3. 更新 process_video 函数
- 增加 `use_pyav` 参数，允许用户选择使用 pyav 或 OpenCV
- 基于参数选择相应的处理器和写入器
- 保持向后兼容性

### 4. 更新依赖文件
- 在 requirements.txt 中添加 pyav 作为可选依赖
- 在 setup.py 中添加 pyav 到 extras_require
- 提供安装说明

### 5. 添加示例代码
- 创建 `12_pyav_video_processing.py` 示例
- 展示如何使用 pyav 进行视频处理
- 比较 pyav 和 OpenCV 的性能

## 技术优势
- **更高的性能**：pyav 基于 FFmpeg，通常比 OpenCV 更快
- **更多的编解码器支持**：支持几乎所有常见的视频编解码器
- **硬件加速**：支持 GPU 加速，进一步提高性能
- **更灵活的格式支持**：支持更多的视频容器格式
- **向后兼容**：保持与现有代码的兼容性

## 实现步骤
1. 创建 `PyAVVideoProcessor` 类
2. 创建 `PyAVVideoWriter` 类
3. 更新 `process_video` 函数
4. 更新依赖文件
5. 添加示例代码
6. 测试和优化

## 预期结果
- 提供更高效的视频处理选项
- 保持与现有代码的兼容性
- 为用户提供选择最适合其需求的视频处理后端的能力
- 提高项目的整体性能和灵活性