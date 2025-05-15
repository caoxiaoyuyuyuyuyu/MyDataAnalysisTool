from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QGroupBox, QPushButton,
                             QLabel, QComboBox, QScrollArea, QHBoxLayout)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QFont


class ControlPanel(QWidget):
    """控制面板组件 - 添加预处理数据控制功能"""
    load_data_clicked = pyqtSignal()
    analyze_clicked = pyqtSignal()
    plot_clicked = pyqtSignal(str, str, str)  # plot_type, x_col, y_col

    def __init__(self):
        super().__init__()
        self.use_processed_btn = None  # 初始化按钮引用
        self.restore_btn = None
        self._setup_ui()

    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # ========== 数据控制部分 ==========
        data_control_group = QGroupBox("数据控制")
        data_control_layout = QVBoxLayout()

        # 使用预处理数据按钮
        self.use_processed_btn = QPushButton("使用预处理数据")
        self.use_processed_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                background-color: #5dade2;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
        """)
        self.use_processed_btn.setEnabled(False)
        data_control_layout.addWidget(self.use_processed_btn)

        # 恢复原始数据按钮
        self.restore_btn = QPushButton("恢复原始数据")
        self.restore_btn.setStyleSheet("""
            QPushButton {
                padding: 8px;
                background-color: #f5b041;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e67e22;
            }
        """)
        self.restore_btn.setEnabled(False)
        data_control_layout.addWidget(self.restore_btn)

        data_control_group.setLayout(data_control_layout)
        layout.addWidget(data_control_group)

        # ========== 其余原有UI代码 ==========
        # 数据操作部分
        load_group = QGroupBox("数据操作")
        load_layout = QVBoxLayout()

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
        plot_layout = QVBoxLayout()

        plot_layout.addWidget(QLabel("图表类型:"))
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["散点图", "折线图", "柱状图", "箱线图"])
        plot_layout.addWidget(self.plot_type_combo)

        plot_layout.addWidget(QLabel("X轴:"))
        self.x_col_combo = QComboBox()
        plot_layout.addWidget(self.x_col_combo)

        plot_layout.addWidget(QLabel("Y轴:"))
        self.y_col_combo = QComboBox()
        plot_layout.addWidget(self.y_col_combo)

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

        layout.addStretch()

    def set_columns(self, columns):
        """设置可选的列名"""
        self.x_col_combo.clear()
        self.y_col_combo.clear()

        self.x_col_combo.addItems(columns)
        self.y_col_combo.addItems(columns)

        if len(columns) >= 2:
            self.x_col_combo.setCurrentIndex(0)
            self.y_col_combo.setCurrentIndex(1)

    def set_processed_btn_state(self, enabled):
        """设置预处理按钮状态"""
        self.use_processed_btn.setEnabled(enabled)

    def set_restore_btn_state(self, enabled):
        """设置恢复按钮状态"""
        self.restore_btn.setEnabled(enabled)

    def _on_plot_clicked(self):
        """处理绘图按钮点击"""
        plot_type = self.plot_type_combo.currentText()
        x_col = self.x_col_combo.currentText()
        y_col = self.y_col_combo.currentText()

        self.plot_clicked.emit(plot_type, x_col, y_col)