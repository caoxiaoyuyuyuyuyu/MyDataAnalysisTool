from PyQt5.QtWidgets import QWidget, QVBoxLayout
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
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            if plot_type == "散点图":
                ax.scatter(data[x_col], data[y_col], alpha=0.6)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{x_col} vs {y_col} 散点图")
                ax.grid(True, linestyle='--', alpha=0.6)

            elif plot_type == "折线图":
                ax.plot(data[x_col], data[y_col], marker='o', linestyle='-')
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{x_col} vs {y_col} 折线图")
                ax.grid(True, linestyle='--', alpha=0.6)

            elif plot_type == "柱状图":
                if data[x_col].dtype == np.object_:
                    # 分类数据
                    data.groupby(x_col)[y_col].mean().plot(kind='bar', ax=ax)
                else:
                    # 数值数据
                    data[y_col].plot(kind='bar', ax=ax)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} 柱状图")
                ax.grid(True, linestyle='--', alpha=0.6)

            elif plot_type == "箱线图":
                data[[x_col, y_col]].boxplot(ax=ax)
                ax.set_title(f"{x_col} 和 {y_col} 箱线图")
                ax.grid(True, linestyle='--', alpha=0.6)

            self.figure.tight_layout()
            self.canvas.draw()

        except Exception as e:
            raise ValueError(f"绘图错误: {str(e)}")