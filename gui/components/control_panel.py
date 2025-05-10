from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton,
                             QLabel, QComboBox, QScrollArea)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont


class ControlPanel(QWidget):
    """控制面板组件"""
    load_data_clicked = pyqtSignal()
    analyze_clicked = pyqtSignal()
    plot_clicked = pyqtSignal(str, str, str)  # plot_type, x_col, y_col

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 数据加载部分
        load_group = QGroupBox("数据操作")
        load_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        load_layout = QVBoxLayout()
        load_layout.setSpacing(8)

        self.load_btn = QPushButton("加载数据文件")
        self.load_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-weight: bold;
            }
        """)
        self.load_btn.clicked.connect(self.load_data_clicked.emit)

        self.analyze_btn = QPushButton("分析数据")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-weight: bold;
            }
        """)
        self.analyze_btn.clicked.connect(self.analyze_clicked.emit)

        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.analyze_btn)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # 图表控制部分
        plot_group = QGroupBox("图表设置")
        plot_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        plot_layout = QVBoxLayout()
        plot_layout.setSpacing(8)

        # 图表类型选择
        plot_layout.addWidget(QLabel("图表类型:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["散点图", "折线图", "柱状图", "箱线图"])
        plot_layout.addWidget(self.plot_type_combo)

        # X轴选择
        plot_layout.addWidget(QLabel("X轴:"))
        self.x_col_combo = QComboBox()
        plot_layout.addWidget(self.x_col_combo)

        # Y轴选择
        plot_layout.addWidget(QLabel("Y轴:"))
        self.y_col_combo = QComboBox()
        plot_layout.addWidget(self.y_col_combo)

        # 绘图按钮
        self.plot_btn = QPushButton("绘制图表")
        self.plot_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.plot_btn.clicked.connect(self._on_plot_clicked)
        plot_layout.addWidget(self.plot_btn)

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # 添加伸缩空间使各部分不会拉伸
        layout.addStretch()

    def set_columns(self, columns):
        """设置可选的列名"""
        self.x_col_combo.clear()
        self.y_col_combo.clear()

        self.x_col_combo.addItems(columns)
        self.y_col_combo.addItems(columns)

        # 默认选择前两列
        if len(columns) >= 2:
            self.x_col_combo.setCurrentIndex(0)
            self.y_col_combo.setCurrentIndex(1)

    def _on_plot_clicked(self):
        """处理绘图按钮点击"""
        plot_type = self.plot_type_combo.currentText()
        x_col = self.x_col_combo.currentText()
        y_col = self.y_col_combo.currentText()

        self.plot_clicked.emit(plot_type, x_col, y_col)