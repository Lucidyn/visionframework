"""
综合测试脚本 - 运行所有测试
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_test_module(module_name, description):
    """运行测试模块"""
    print("\n" + "=" * 70)
    print(f"运行测试: {description}")
    print("=" * 70)
    
    try:
        # 动态导入测试模块
        test_module = __import__(f"tests.{module_name}", fromlist=[module_name])
        
        # 如果模块有 main 函数，运行它
        if hasattr(test_module, 'main'):
            result = test_module.main()
            return result
        else:
            print(f"[SKIP] {module_name} 没有 main() 函数")
            return None
    except Exception as e:
        print(f"[ERROR] 运行 {module_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("=" * 70)
    print("Vision Framework - 综合测试套件")
    print("=" * 70)
    print("\n本测试将运行所有可用的测试模块")
    print("=" * 70)
    
    # 定义测试模块列表
    test_modules = [
        ("test_structure", "项目结构测试"),
        ("quick_test", "快速功能测试"),
        ("test_utilities", "工具和组件测试"),
        ("test_rfdetr", "RF-DETR 检测器测试"),
        ("test_logging_exceptions", "日志和异常处理测试"),
        ("test_code_quality", "代码质量测试（配置验证、类型提示、文档）"),
    ]
    
    results = {}
    
    # 运行每个测试模块
    for module_name, description in test_modules:
        result = run_test_module(module_name, description)
        results[module_name] = result
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = []
    failed = []
    skipped = []
    
    for module_name, result in results.items():
        if result is True:
            passed.append(module_name)
            print(f"[✓] {module_name}: 通过")
        elif result is False:
            failed.append(module_name)
            print(f"[✗] {module_name}: 失败")
        else:
            skipped.append(module_name)
            print(f"[○] {module_name}: 跳过")
    
    print("\n" + "=" * 70)
    print(f"总计: {len(passed)} 通过, {len(failed)} 失败, {len(skipped)} 跳过")
    print("=" * 70)
    
    if len(failed) == 0:
        if len(passed) > 0:
            print("\n[✓] 所有可用的测试都已通过！")
        else:
            print("\n[○] 没有可运行的测试（可能缺少依赖）")
    else:
        print(f"\n[✗] 有 {len(failed)} 个测试失败，请检查上面的错误信息")
    
    print("\n提示:")
    print("  - 如果某些测试被跳过，可能是因为缺少相应的依赖")
    print("  - 安装所有依赖: pip install -r requirements.txt")
    print("  - RF-DETR 需要额外安装: pip install rfdetr supervision")
    print("=" * 70)
    
    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

