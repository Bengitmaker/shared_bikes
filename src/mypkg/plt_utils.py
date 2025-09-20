# plt_utils.py（工具模块）
import matplotlib.pyplot as plt

def set_chinese_font():
    """设置matplotlib中文字体（调用一次即可）"""
    try:
        # 按系统适配字体（也可根据需求固定某一种）
        import platform
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'DejaVu Sans']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"字体配置失败：{e}")

# 可选：自动执行配置（导入模块时直接生效，无需手动调用）
set_chinese_font()