from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QStatusBar, QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt
from gui.components.control_panel import ControlPanel
from gui.components.chart_panel import ChartPanel
from gui.components.analysis_panel import AnalysisPanel
from core.data_loader import DataLoader
from core.analyzer import DataAnalyzer


class MainWindow(QMainWindow):
    """主窗口类 - 上下布局"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("数据分析可视化工具")
        self.setGeometry(100, 100, 1200, 900)

        # 初始化核心组件
        self.data_loader = DataLoader()
        self.analyzer = DataAnalyzer()
        self.current_data = None

        # 设置UI
        self._setup_ui()

        # 连接信号槽
        self._connect_signals()

    def _setup_ui(self):
        """初始化用户界面 - 上下布局"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主垂直布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # 上部：数据分析面板 (40%高度)
        self.analysis_panel = AnalysisPanel()
        main_layout.addWidget(self.analysis_panel, stretch=4)

        # 下部：控制面板+图表区域 (60%高度)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(5)

        # 左侧图表区域 (75%宽度)
        self.chart_panel = ChartPanel()
        bottom_layout.addWidget(self.chart_panel, stretch=6)

        # 右侧控制面板 (25%宽度)
        self.control_panel = ControlPanel()
        bottom_layout.addWidget(self.control_panel, stretch=2)

        main_layout.addWidget(bottom_widget, stretch=6)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪", 3000)

    def _connect_signals(self):
        """连接信号和槽"""
        # 控制面板信号
        self.control_panel.load_data_clicked.connect(self._load_data)
        self.control_panel.analyze_clicked.connect(self._analyze_data)
        self.control_panel.plot_clicked.connect(self._plot_data)

        # 数据加载器信号
        self.data_loader.data_loaded.connect(self._on_data_loaded)

    def _load_data(self):
        """加载数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开数据文件", "",
            "数据文件 (*.csv *.xls *.xlsx);;所有文件 (*.*)"
        )

        if file_path:
            try:
                self.data_loader.load_file(file_path)
                self.status_bar.showMessage(f"已加载文件: {file_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文件失败: {str(e)}")

    def _on_data_loaded(self, data):
        """数据加载完成处理"""
        self.current_data = data
        self.control_panel.set_columns(data.columns.tolist())
        self.status_bar.showMessage(f"已加载数据: {len(data)}行×{len(data.columns)}列", 3000)

    def _analyze_data(self):
        """分析数据"""
        if self.current_data is None:
            QMessageBox.warning(self, "警告", "请先加载数据文件")
            return

        try:
            # 获取各种分析结果
            stats = self.analyzer.describe_data(self.current_data)
            dtypes = self.analyzer.get_column_types(self.current_data)
            unique_counts = self.analyzer.get_unique_counts(self.current_data)
            missing_values = self.analyzer.get_missing_values(self.current_data)

            # 将分析结果显示在分析面板
            self.analysis_panel.show_analysis_results({
                '统计量': stats,
                '数据类型': dtypes,
                '唯一值数量': unique_counts,
                '缺失值数量': missing_values
            })
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据分析失败: {str(e)}")

    def _plot_data(self, plot_type, x_col, y_col):
        """绘制图表"""
        if self.current_data is None:
            QMessageBox.warning(self, "警告", "请先加载数据文件")
            return

        try:
            self.chart_panel.plot_data(
                self.current_data, plot_type, x_col, y_col
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"绘图失败: {str(e)}")