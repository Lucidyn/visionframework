#!/usr/bin/env python3
"""
依赖管理使用示例

本示例演示如何使用 Vision Framework 的依赖管理功能。
包括：
1. 创建和使用 DependencyManager
2. 检查依赖可用性
3. 导入可选依赖
4. 使用延迟导入装饰器
5. 获取依赖信息和安装命令
6. 依赖状态管理
7. 依赖管理最佳实践
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionframework.utils.dependency_manager import (
    DependencyManager,
    dependency_manager,
    is_dependency_available,
    get_available_dependencies,
    get_missing_dependencies,
    validate_dependency,
    get_install_command,
    import_optional_dependency,
    lazy_import
)

def test_basic_dependency_management():
    """
    测试基本的依赖管理功能
    """
    print("=== 测试基本依赖管理功能 ===")
    
    # 1. 创建依赖管理器
    print("\n1. 创建依赖管理器:")
    manager = DependencyManager()
    print("依赖管理器创建成功")
    
    # 2. 检查依赖可用性
    print("\n2. 检查依赖可用性:")
    dependencies = ["clip", "sam", "rfdetr", "pyav", "dev"]
    for dep in dependencies:
        available = is_dependency_available(dep)
        print(f"{dep}: {available}")
    
    # 3. 获取可用和缺失的依赖
    print("\n3. 获取可用和缺失的依赖:")
    available = get_available_dependencies()
    print(f"可用的依赖: {available}")
    
    missing = get_missing_dependencies()
    print(f"缺失的依赖: {missing}")
    
    # 4. 获取依赖信息
    print("\n4. 获取依赖信息:")
    for dep in dependencies:
        info = manager.get_dependency_info(dep)
        if info:
            print(f"{dep} 依赖信息:")
            print(f"  包: {info.get('packages', [])}")
            print(f"  最低版本: {info.get('minimum_version', 'N/A')}")
            print(f"  描述: {info.get('description', 'N/A')}")
        else:
            print(f"{dep} 依赖信息: 不存在")
    
    # 5. 获取安装命令
    print("\n5. 获取安装命令:")
    for dep in dependencies:
        command = get_install_command(dep)
        if command:
            print(f"{dep} 安装命令: {command}")
        else:
            print(f"{dep} 安装命令: 无")
    
    return manager

def test_dependency_import():
    """
    测试依赖导入功能
    """
    print("\n=== 测试依赖导入功能 ===")
    
    manager = DependencyManager()
    
    # 1. 测试导入可用依赖
    print("\n1. 测试导入可用依赖:")
    # 测试导入 numpy (核心依赖，应该可用)
    try:
        import numpy as np
        print("numpy 导入成功")
    except ImportError as e:
        print(f"numpy 导入失败: {e}")
    
    # 2. 测试导入可选依赖
    print("\n2. 测试导入可选依赖:")
    # 测试导入 transformers (clip 依赖)
    print("测试导入 transformers (clip 依赖):")
    transformers_module = import_optional_dependency("clip", "transformers")
    if transformers_module:
        print("transformers 导入成功")
        print(f"transformers 版本: {getattr(transformers_module, '__version__', 'N/A')}")
    else:
        print("transformers 导入失败 (可能未安装)")
    
    # 测试导入 av (pyav 依赖)
    print("\n测试导入 av (pyav 依赖):")
    av_module = import_optional_dependency("pyav", "av")
    if av_module:
        print("av 导入成功")
        print(f"av 版本: {getattr(av_module, '__version__', 'N/A')}")
    else:
        print("av 导入失败 (可能未安装)")
    
    # 3. 测试依赖验证
    print("\n3. 测试依赖验证:")
    for dep in ["clip", "pyav"]:
        valid = validate_dependency(dep)
        print(f"{dep} 依赖验证: {'成功' if valid else '失败'}")
    
    return manager

def test_lazy_import():
    """
    测试延迟导入功能
    """
    print("\n=== 测试延迟导入功能 ===")
    
    # 1. 使用延迟导入装饰器
    print("\n1. 使用延迟导入装饰器:")
    
    @lazy_import("clip", "transformers")
    def test_clip_function():
        """
        测试使用 CLIP 依赖的函数
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
            print("成功导入 CLIP 相关模块")
            return True
        except ImportError as e:
            print(f"CLIP 模块导入失败: {e}")
            return False
    
    @lazy_import("pyav", "av")
    def test_pyav_function():
        """
        测试使用 PyAV 依赖的函数
        """
        try:
            import av
            print("成功导入 PyAV 模块")
            return True
        except ImportError as e:
            print(f"PyAV 模块导入失败: {e}")
            return False
    
    # 2. 测试延迟导入函数
    print("\n2. 测试延迟导入函数:")
    print("测试 CLIP 函数:")
    result = test_clip_function()
    print(f"CLIP 函数执行结果: {result}")
    
    print("\n测试 PyAV 函数:")
    result = test_pyav_function()
    print(f"PyAV 函数执行结果: {result}")
    
    return True

def test_dependency_status():
    """
    测试依赖状态管理
    """
    print("\n=== 测试依赖状态管理 ===")
    
    manager = DependencyManager()
    
    # 1. 获取单个依赖状态
    print("\n1. 获取单个依赖状态:")
    for dep in ["clip", "sam", "rfdetr", "pyav", "dev"]:
        status = manager.get_dependency_status(dep)
        print(f"{dep} 依赖状态:")
        print(f"  可用: {status.get('available', False)}")
        print(f"  消息: {status.get('message', 'N/A')}")
    
    # 2. 获取所有依赖状态
    print("\n2. 获取所有依赖状态:")
    all_status = manager.get_all_dependency_status()
    for dep, status in all_status.items():
        print(f"{dep}: {status['available']} - {status['message']}")
    
    # 3. 测试依赖状态缓存
    print("\n3. 测试依赖状态缓存:")
    # 第一次检查（应该触发实际检查）
    start_time = time.time()
    available1 = is_dependency_available("clip")
    time1 = time.time() - start_time
    print(f"第一次检查 clip 依赖: {available1}, 耗时: {time1:.4f}秒")
    
    # 第二次检查（应该使用缓存）
    start_time = time.time()
    available2 = is_dependency_available("clip")
    time2 = time.time() - start_time
    print(f"第二次检查 clip 依赖: {available2}, 耗时: {time2:.4f}秒")
    
    print(f"缓存效果: {time2 < time1 * 0.1}")  # 第二次应该快很多
    
    return manager

def test_dependency_best_practices():
    """
    测试依赖管理最佳实践
    """
    print("\n=== 测试依赖管理最佳实践 ===")
    
    # 1. 条件导入示例
    print("\n1. 条件导入示例:")
    
    def process_with_optional_features(image):
        """
        处理图像，使用可选功能（如果可用）
        """
        print("基础图像处理...")
        
        # 检查并使用 CLIP 功能（如果可用）
        if is_dependency_available("clip"):
            print("CLIP 依赖可用，使用 CLIP 功能...")
            transformers_module = import_optional_dependency("clip", "transformers")
            if transformers_module:
                print("成功使用 CLIP 功能")
        else:
            print("CLIP 依赖不可用，跳过 CLIP 功能")
        
        # 检查并使用 PyAV 功能（如果可用）
        if is_dependency_available("pyav"):
            print("PyAV 依赖可用，使用 PyAV 功能...")
            av_module = import_optional_dependency("pyav", "av")
            if av_module:
                print("成功使用 PyAV 功能")
        else:
            print("PyAV 依赖不可用，跳过 PyAV 功能")
        
        return True
    
    # 测试条件导入函数
    print("测试条件导入函数:")
    result = process_with_optional_features("dummy_image")
    print(f"条件导入函数执行结果: {result}")
    
    # 2. 依赖管理装饰器示例
    print("\n2. 依赖管理装饰器示例:")
    
    def requires_dependency(dependency):
        """
        依赖检查装饰器
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                if not is_dependency_available(dependency):
                    print(f"错误: 需要 {dependency} 依赖")
                    install_cmd = get_install_command(dependency)
                    if install_cmd:
                        print(f"请运行以下命令安装: {install_cmd}")
                    return None
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    @requires_dependency("clip")
    def clip_related_function():
        """
        需要 CLIP 依赖的函数
        """
        print("执行 CLIP 相关函数")
        return "CLIP 函数执行成功"
    
    @requires_dependency("pyav")
    def pyav_related_function():
        """
        需要 PyAV 依赖的函数
        """
        print("执行 PyAV 相关函数")
        return "PyAV 函数执行成功"
    
    # 测试依赖检查装饰器
    print("测试 CLIP 相关函数:")
    result = clip_related_function()
    print(f"CLIP 函数执行结果: {result}")
    
    print("\n测试 PyAV 相关函数:")
    result = pyav_related_function()
    print(f"PyAV 函数执行结果: {result}")
    
    # 3. 依赖状态报告
    print("\n3. 依赖状态报告:")
    
    def generate_dependency_report():
        """
        生成依赖状态报告
        """
        manager = DependencyManager()
        available = get_available_dependencies()
        missing = get_missing_dependencies()
        
        print("=== 依赖状态报告 ===")
        print(f"可用依赖: {len(available)}")
        for dep in available:
            info = manager.get_dependency_info(dep)
            print(f"  - {dep}: {info.get('description', 'N/A')}")
        
        print(f"\n缺失依赖: {len(missing)}")
        for dep in missing:
            info = manager.get_dependency_info(dep)
            install_cmd = get_install_command(dep)
            print(f"  - {dep}: {info.get('description', 'N/A')}")
            if install_cmd:
                print(f"    安装命令: {install_cmd}")
        
        print("=== 报告结束 ===")
    
    generate_dependency_report()
    
    return True

def test_global_dependency_manager():
    """
    测试全局依赖管理器
    """
    print("\n=== 测试全局依赖管理器 ===")
    
    # 1. 使用全局依赖管理器
    print("\n1. 使用全局依赖管理器:")
    from visionframework.utils.dependency_manager import dependency_manager
    
    print("全局依赖管理器使用成功")
    
    # 2. 测试全局管理器功能
    print("\n2. 测试全局管理器功能:")
    # 检查依赖可用性
    clip_available = dependency_manager.is_available("clip")
    print(f"全局管理器 - clip 依赖可用: {clip_available}")
    
    # 获取依赖信息
    sam_info = dependency_manager.get_dependency_info("sam")
    print(f"全局管理器 - sam 依赖信息: {sam_info}")
    
    # 3. 测试全局函数
    print("\n3. 测试全局函数:")
    # 使用全局函数检查依赖
    pyav_available = is_dependency_available("pyav")
    print(f"全局函数 - pyav 依赖可用: {pyav_available}")
    
    # 获取可用依赖列表
    available_deps = get_available_dependencies()
    print(f"全局函数 - 可用依赖: {available_deps}")
    
    return True

def main():
    """
    主函数
    """
    print("=== 依赖管理示例 ===")
    
    try:
        # 测试基本依赖管理功能
        basic_manager = test_basic_dependency_management()
        
        # 测试依赖导入功能
        import_manager = test_dependency_import()
        
        # 测试延迟导入功能
        lazy_import_result = test_lazy_import()
        
        # 测试依赖状态管理
        status_manager = test_dependency_status()
        
        # 测试依赖管理最佳实践
        best_practices_result = test_dependency_best_practices()
        
        # 测试全局依赖管理器
        global_manager_result = test_global_dependency_manager()
        
        print("\n=== 依赖管理示例完成 ===")
        print("所有测试通过！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
