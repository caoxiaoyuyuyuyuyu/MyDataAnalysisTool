import pandas as pd
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QStatusBar, QFileDialog, QMessageBox,
                             QMenuBar, QAction)
from PyQt5.QtCore import Qt, pyqtSignal
from gui.components.control_panel import ControlPanel
from gui.components.chart_panel import ChartPanel
from gui.components.analysis_panel import AnalysisPanel
from core.data_loader import DataLoader
from core.analyzer import DataAnalyzer
from gui.predict_window import PredictWindow


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
        self.processed_data = None
        self.current_model = None
        self.current_metrics = None

        self.original_data = None  # 保存原始数据备份


        # 添加数据指纹和历史记录存储
        self.current_data_fingerprint = None
        self.training_history = {}

        # 设置UI
        self._setup_menus()  # 先设置菜单栏
        self._setup_ui()

        # 连接信号槽
        self._connect_signals()

    def _setup_menus(self):
        """设置菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        open_action = QAction("打开数据文件", self)
        open_action.triggered.connect(self._load_data)  # 连接现有的_load_data方法
        file_menu.addAction(open_action)

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 数据菜单
        data_menu = menubar.addMenu("数据")

        preprocess_action = QAction("数据预处理", self)
        preprocess_action.triggered.connect(self._open_preprocess_window)
        data_menu.addAction(preprocess_action)

        # 模型菜单
        model_menu = menubar.addMenu("模型")

        train_action = QAction("模型训练", self)
        train_action.triggered.connect(self._open_train_window)
        model_menu.addAction(train_action)

        compare_action = QAction("高级模型训练", self)
        compare_action.triggered.connect(self._open_advanced_train_window)
        model_menu.addAction(compare_action)

        # 预测菜单
        predict_menu = menubar.addMenu("预测")
        predict_action = QAction("模型预测", self)
        predict_action.triggered.connect(self._open_predict_window)
        predict_menu.addAction(predict_action)

    def _open_predict_window(self):
        """打开预测窗口"""
        X = self.processed_data.iloc[:, :-1]
        # 目标列名
        target = self.processed_data.columns[-1]
        predict_window = PredictWindow(X, target, self.training_history[self.current_data_fingerprint], self)
        predict_window.exec_()

    def _setup_ui(self):
        """初始化用户界面 - 使用QSplitter实现可调整的上下布局"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主垂直布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # 创建垂直分割器
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(5)  # 设置分隔条宽度
        splitter.setStyleSheet("""
            QSplitter::handle {
                background: #ccc;
            }
        """)

        # 上部：数据分析面板 (初始比例40%)
        self.analysis_panel = AnalysisPanel()
        splitter.addWidget(self.analysis_panel)

        # 下部：控制面板+图表区域 (初始比例60%)
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

        splitter.addWidget(bottom_widget)

        # 设置初始比例 (40:60)
        splitter.setSizes([400, 600])  # 根据窗口大小自动按比例分配

        main_layout.addWidget(splitter)

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

        self.control_panel.use_processed_btn.clicked.connect(self._use_processed_data)
        self.control_panel.restore_btn.clicked.connect(self._restore_original_data)

    def _load_data(self):
        """加载数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开数据文件", "",
            "数据文件 (*.csv *.xls *.xlsx *.txt);;文本文件 (*.txt);;所有文件 (*.*)"
        )

        if file_path:
            try:
                self.data_loader.load_file(file_path)
                self.status_bar.showMessage(f"已加载文件: {file_path}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文件失败: {str(e)}")

    def _on_data_loaded(self, data):
        try:
            """数据加载完成处理"""
            if data.empty:
                QMessageBox.warning(self, "警告", "加载的数据为空")
                return

            # 获取首行数据用于展示
            first_row = data.iloc[0].to_dict()
            total_columns = len(first_row)

            # 构建列显示范围
            if total_columns <= 10:
                display_columns = list(range(total_columns))
            else:
                display_columns = list(range(5)) + list(range(total_columns - 5, total_columns))

            # 格式化显示内容
            msg_lines = []
            for idx, col_idx in enumerate(display_columns):
                col_name = data.columns[col_idx]
                value = first_row[col_name]

                # 添加省略号提示
                if idx == 5 and total_columns > 10:
                    msg_lines.append(f"......[中间省略{total_columns - 10}列]......")

                msg_lines.append(f"列{col_idx + 1} ({col_name}): {str(value)[:50]}")  # 限制值显示长度

            # 构建提示消息
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("列名处理")
            msg.setText("检测到数据首行内容：\n" + "\n".join(msg_lines))

            # 添加自定义按钮
            use_header_btn = msg.addButton("使用首行为列名", QMessageBox.ActionRole)
            generate_btn = msg.addButton("生成新列名", QMessageBox.ActionRole)
            cancel_btn = msg.addButton("取消", QMessageBox.RejectRole)

            msg.exec_()

            # 处理用户选择
            clicked = msg.clickedButton()
            if clicked == generate_btn:
                data = self._auto_add_column_names(data)
            elif clicked == cancel_btn:
                data = pd.DataFrame()
                self.status_bar.showMessage("数据加载已取消", 3000)
                return

            if data.empty:
                QMessageBox.warning(self, "警告", "数据为空，请重新加载")
                return

            # 更新数据状态
            self.current_data = data
            self.original_data = None
            self.processed_data = None

            # 更新界面组件
            self.control_panel.set_columns(data.columns.tolist())
            self.control_panel.set_processed_btn_state(False)
            self.control_panel.restore_btn.setEnabled(False)
            self.status_bar.showMessage(f"已加载数据: {len(data)}行×{len(data.columns)}列", 3000)
            self.current_data_fingerprint = self._get_data_fingerprint(data)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据加载失败: {str(e)}")

    def _auto_add_column_names(self, data):
        """生成格式化的列名（列1, 列2...）"""
        new_columns = [f'Column{i + 1}' for i in range(data.shape[1])]
        return data.rename(columns=dict(zip(data.columns, new_columns)))

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
            self.analysis_panel.set_current_data(self.current_data)

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

    def _open_preprocess_window(self):
        """打开预处理窗口"""
        if self.current_data is None:
            QMessageBox.warning(self, "警告", "请先加载数据文件")
            return

        # 确保窗口不会被垃圾回收
        if hasattr(self, 'preprocess_window'):
            self.preprocess_window.close()

        from gui.preprocess_window import PreprocessWindow
        self.preprocess_window = PreprocessWindow(self.current_data, self)
        self.preprocess_window.setAttribute(Qt.WA_DeleteOnClose, False)  # 重要！
        self.preprocess_window.preprocessing_done.connect(self._on_preprocessing_done)
        self.preprocess_window.show()

    def _on_model_trained(self, result):
        """模型训练完成处理"""
        self.current_model = result['model']
        self.current_metrics = result['metrics']
        QMessageBox.information(self, "成功", "模型训练完成")

    def _open_advanced_train_window(self):
        """打开高级模型训练窗口"""
        if not hasattr(self, 'processed_data') or self.processed_data is None:
            QMessageBox.warning(self, "警告", "请先预处理数据")
            return

        # 生成数据指纹
        X = self.processed_data.iloc[:, :-1]
        y = self.processed_data.iloc[:, -1]
        current_fingerprint = self._get_data_fingerprint(pd.concat([X, y], axis=1))

        # 传递历史记录引用和指纹给训练窗口
        from gui.advanced_train_window import AdvancedTrainWindow
        self.advanced_train_window = AdvancedTrainWindow(
            X, y,
            parent=self,
            data_fingerprint=current_fingerprint,
            history=self.training_history.setdefault(current_fingerprint, [])
        )
        self.advanced_train_window.model_trained.connect(self._on_model_trained)
        self.advanced_train_window.exec_()

    def _on_preprocessing_done(self, processed_data):
        """保存预处理数据并更新按钮状态"""
        try:
            # 保存预处理结果
            self.processed_data = processed_data

            # 备份原始数据(如果尚未备份)
            if self.original_data is None:
                self.original_data = self.current_data.copy()

            # 启用"使用预处理数据"按钮
            self.control_panel.set_processed_btn_state(True)

            QMessageBox.information(
                self,
                "预处理完成",
                "数据预处理已完成，请点击'使用预处理数据'按钮应用更改"
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存预处理数据失败: {str(e)}")

    # 添加新方法：使用预处理数据
    def _use_processed_data(self):
        """切换到预处理后的数据"""
        if hasattr(self, 'processed_data') and self.processed_data is not None:
            # 更新当前数据
            self.current_data = self.processed_data

            # 更新控制面板
            self.control_panel.set_columns(self.processed_data.columns.tolist())
            self.control_panel.set_processed_btn_state(False)  # 禁用按钮
            self.control_panel.set_restore_btn_state(True)

            # 刷新分析结果
            self._analyze_data()

            # 重置图表
            self.chart_panel.figure.clear()
            self.chart_panel.canvas.draw()

            self.status_bar.showMessage("已切换到预处理数据", 3000)
            # 更新数据指纹
            self.current_data_fingerprint = self._get_data_fingerprint(self.processed_data)


        else:
            QMessageBox.warning(self, "警告", "没有可用的预处理数据")

    def _restore_original_data(self):
        """恢复原始数据"""
        if self.original_data is not None:
            # 恢复数据
            self.current_data = self.original_data

            # 更新控制面板
            self.control_panel.set_columns(self.original_data.columns.tolist())
            self.control_panel.set_processed_btn_state(True)  # 启用"使用预处理数据"按钮
            self.control_panel.set_restore_btn_state(False)  # 禁用恢复按钮

            # 刷新分析结果
            self._analyze_data()

            # 重置图表
            self.chart_panel.figure.clear()
            self.chart_panel.canvas.draw()

            self.status_bar.showMessage("已恢复原始数据", 3000)
            # 更新数据指纹
            self.current_data_fingerprint = self._get_data_fingerprint(self.original_data)
    def _get_data_fingerprint(self, data):
        """生成数据唯一标识"""
        try:
            # 使用pandas的哈希函数生成指纹
            hashes = pd.util.hash_pandas_object(data, index=True).values
            return hash(tuple(hashes))
        except AttributeError:
            # pandas旧版本兼容
            return hash(data.to_csv().encode())

    def _open_train_window(self):
        """打开模型训练窗口"""
        if not hasattr(self, 'processed_data') or self.processed_data is None:
            QMessageBox.warning(self, "警告", "请先预处理数据")
            return

        # 生成数据指纹
        X = self.processed_data.iloc[:, :-1]
        y = self.processed_data.iloc[:, -1]
        current_fingerprint = self._get_data_fingerprint(pd.concat([X, y], axis=1))

        # 传递历史记录引用和指纹给训练窗口
        from gui.train_window import TrainWindow
        self.train_window = TrainWindow(
            X, y,
            parent=self,
            data_fingerprint=current_fingerprint,
            history=self.training_history.setdefault(current_fingerprint, [])
        )
        self.train_window.model_trained.connect(self._on_model_trained)
        self.train_window.exec_()
