#!/usr/bin/env python3
"""
RK3588开发板专用启动脚本
适用于瑞芯微RK3588平台，优化了性能和兼容性
"""
import sys
import os
import subprocess

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_environment():
    """检查运行环境"""
    print("=== 检查运行环境 ===")
    
    # 检查是否在RK3588平台
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
        if 'RK3588' in cpuinfo:
            print("✓ 检测到RK3588平台")
        else:
            print("⚠ 未检测到RK3588平台，可能影响性能")
    except:
        print("⚠ 无法检测CPU信息")
    
    # 检查Python版本
    print(f"✓ Python版本: {sys.version}")
    
    # 检查依赖库
    required_libs = ['PyQt6', 'cv2', 'numpy']
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✓ 已安装 {lib}")
        except ImportError:
            print(f"✗ 未安装 {lib}")
            return False
    
    print("=== 环境检查完成 ===")
    return True

def optimize_performance():
    """优化RK3588性能"""
    print("=== 优化性能设置 ===")
    
    try:
        # 设置CPU性能模式
        subprocess.run(['echo', 'performance'], stdout=open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', 'w'), check=True)
        print("✓ CPU设置为性能模式")
    except:
        print("⚠ 无法设置CPU性能模式")
    
    try:
        # 禁用屏幕保护
        subprocess.run(['xset', 's', 'off'], check=True)
        subprocess.run(['xset', '-dpms'], check=True)
        print("✓ 已禁用屏幕保护")
    except:
        print("⚠ 无法禁用屏幕保护")
    
    print("=== 性能优化完成 ===")

def main():
    """主函数"""
    print("=== 负压爬壁机器人控制系统 ===")
    print("=== RK3588开发板专用版本 ===")
    
    # 检查环境
    if not check_environment():
        print("环境检查失败，请先安装必要的依赖库")
        sys.exit(1)
    
    # 优化性能
    optimize_performance()
    
    # 启动主程序
    print("=== 启动主程序 ===")
    try:
        import main
        main.main()
    except Exception as e:
        print(f"程序启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()