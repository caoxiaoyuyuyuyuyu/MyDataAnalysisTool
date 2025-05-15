from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pandas as pd
import numpy as np


class ChartPanel(QWidget):
    """图表显示面板"""

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建Matplotlib图形
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # 添加导航工具栏
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

    def plot_data(self, data: pd.DataFrame, plot_type: str, x_col: str, y_col: str):
        """根据选择绘制图表"""
        try:
            # 清空图形并重置设置
            self.figure.clear()
            ax = self.figure.add_subplot(111)

            # 一次性设置字体（避免重复设置）
            plt.rcParams.update({
                'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei'],
                'axes.unicode_minus': False
            })

            if plot_type == "散点图":
                ax.scatter(data[x_col], data[y_col], alpha=0.6)
                ax.set_xlabel(x_col, fontsize=10)
                ax.set_ylabel(y_col, fontsize=10)
                ax.set_title(f"{x_col} vs {y_col} 散点图", fontsize=12)

            elif plot_type == "折线图":
                ax.plot(data[x_col], data[y_col], marker='o', linestyle='-', linewidth=1)
                ax.set_xlabel(x_col, fontsize=10)
                ax.set_ylabel(y_col, fontsize=10)
                ax.set_title(f"{x_col} vs {y_col} 折线图", fontsize=12)

            elif plot_type == "柱状图":
                # 优化柱状图逻辑
                if pd.api.types.is_numeric_dtype(data[x_col]):
                    # 数值型数据分箱处理
                    bins = min(20, data[x_col].nunique())
                    ax.hist2d(data[x_col], data[y_col], bins=(bins, 10), cmap='Blues')
                    ax.set_xlabel(x_col, fontsize=10)
                    ax.set_ylabel(y_col, fontsize=10)
                    ax.set_title(f"{x_col} 与 {y_col} 分布热力图", fontsize=12)
                else:
                    # 分类数据处理
                    top_n = min(15, data[x_col].nunique())  # 最多显示15个类别
                    counts = data[x_col].value_counts().nlargest(top_n)
                    counts.plot.bar(ax=ax, rot=45)
                    ax.set_xlabel(x_col, fontsize=10)
                    ax.set_ylabel("数量", fontsize=10)
                    ax.set_title(f"前 {top_n} 个 {x_col} 类别分布", fontsize=12)

            elif plot_type == "箱线图":
                if pd.api.types.is_numeric_dtype(data[x_col]):
                    data.boxplot(column=y_col, by=x_col, ax=ax)
                    ax.set_title(f"{x_col} 分组下的 {y_col} 箱线图", fontsize=12)
                else:
                    ax.boxplot(data[y_col])
                    ax.set_xticks([1])
                    ax.set_xticklabels([y_col])
                    ax.set_title(f"{y_col} 箱线图", fontsize=12)

            # 统一美化设置
            ax.grid(True, linestyle='--', alpha=0.6)
            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            raise ValueError(f"绘图错误: {str(e)}")