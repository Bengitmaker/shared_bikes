"""
测试运行脚本

提供便捷的方式运行项目测试。
"""

import pytest
import sys
import os
from pathlib import Path


def run_all_tests():
    """运行所有测试"""
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent
    tests_dir = root_dir / "tests"
    
    # 构建pytest参数
    pytest_args = [
        str(tests_dir),
        "-v",  # 详细输出
        "--tb=short",  # 简短的回溯信息
        "-x",  # 遇到第一个失败就停止
        "--capture=no"  # 不捕获输出，便于调试
    ]
    
    print("开始运行测试...")
    print(f"测试目录: {tests_dir}")
    
    # 运行测试
    return_code = pytest.main(pytest_args)
    
    if return_code == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n❌ 测试失败，返回码: {return_code}")
    
    return return_code


def run_unit_tests():
    """只运行单元测试"""
    root_dir = Path(__file__).parent.parent
    unit_tests_dir = root_dir / "tests" / "unit"
    
    pytest_args = [
        str(unit_tests_dir),
        "-v",
        "--tb=short",
        "-x",
        "--capture=no"
    ]
    
    print("开始运行单元测试...")
    return_code = pytest.main(pytest_args)
    
    if return_code == 0:
        print("\n🎉 单元测试通过！")
    else:
        print(f"\n❌ 单元测试失败，返回码: {return_code}")
    
    return return_code


def run_integration_tests():
    """只运行集成测试"""
    root_dir = Path(__file__).parent.parent
    integration_tests_dir = root_dir / "tests" / "integration"
    
    pytest_args = [
        str(integration_tests_dir),
        "-v",
        "--tb=short",
        "-x",
        "--capture=no"
    ]
    
    print("开始运行集成测试...")
    return_code = pytest.main(pytest_args)
    
    if return_code == 0:
        print("\n🎉 集成测试通过！")
    else:
        print(f"\n❌ 集成测试失败，返回码: {return_code}")
    
    return return_code


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "unit":
        result = run_unit_tests()
    elif len(sys.argv) > 1 and sys.argv[1] == "integration":
        result = run_integration_tests()
    else:
        result = run_all_tests()
    
    sys.exit(result)