import pandas as pd
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QComboBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSpinBox,
                             QCheckBox, QMessageBox, QLineEdit, QFormLayout, QTabWidget, QWidget, QScrollArea,
                             QDoubleSpinBox, QListWidgetItem, QListWidget, QSplitter, QApplication, QProgressDialog,
                             QRadioButton)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.model_selection import GridSearchCV

from core.model_trainer import ModelTrainer


class ModelParamsWidget(QWidget):
    """动态模型参数控件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.current_params = {}

    def update_params(self, model_name, problem_type):
        """根据模型更新参数控件"""
        self.clear_params()

        if problem_type == 'regression':
            if model_name == 'Linear Regression':
                self.add_checkbox('拟合截距', 'fit_intercept', True)
            elif model_name == 'Polynomial Regression':
                self.add_spinbox('多项式次数', 'degree', 1, 10, 2)
            elif model_name == 'Ridge Regression':
                self.add_double_spinbox('正则化强度', 'alpha', 0.000001, 100.0, 0.000001)
            elif model_name == 'Lasso Regression':
                self.add_double_spinbox('正则化强度', 'alpha', 0.000001, 100.0, 0.000001)
            elif model_name == 'Decision Tree':
                self.add_spinbox('最大深度', 'max_depth', 1, 50, 5)
                self.add_spinbox('最小样本分割', 'min_samples_split', 2, 20, 2)
            elif model_name == 'Random Forest':
                self.add_spinbox('树的数量', 'n_estimators', 10, 500, 100)
                self.add_spinbox('最大深度', 'max_depth', 1, 50, 5)
            elif model_name == 'SVR':
                self.add_combobox('核函数', 'kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                self.add_double_spinbox('C参数', 'C', 0.1, 100.0, 1.0)
            elif model_name == 'KNN Regression':
                self.add_spinbox('邻居数', 'n_neighbors', 1, 20, 5)

        elif problem_type == 'classification':
            if model_name == 'Logistic Regression':
                self.add_double_spinbox('C参数', 'C', 0.01, 100.0, 1.0)
                self.add_combobox('正则化类型', 'penalty', ['l2', 'none'])
            elif model_name == 'Decision Tree':
                self.add_spinbox('最大深度', 'max_depth', 1, 50, 5)
                self.add_spinbox('最小样本分割', 'min_samples_split', 2, 20, 2)
            elif model_name == 'Random Forest':
                self.add_spinbox('树的数量', 'n_estimators', 10, 500, 100)
                self.add_spinbox('最大深度', 'max_depth', 1, 50, 5)
            elif model_name == 'SVM':
                self.add_combobox('核函数', 'kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
                self.add_double_spinbox('C参数', 'C', 0.1, 100.0, 1.0)
            elif model_name == 'KNN Classification':
                self.add_spinbox('邻居数', 'n_neighbors', 1, 20, 5)

        elif problem_type == 'clustering':
            if model_name == 'K-Means':
                self.add_spinbox('聚类数', 'n_clusters', 2, 20, 5)
            elif model_name == 'PCA':
                self.add_spinbox('主成分数', 'n_components', 1, min(20, len(self.parent().X.columns)), 2)

    def add_spinbox(self, label, param_name, min_val, max_val, default):
        """添加整数参数控件"""
        group = QGroupBox(label)
        layout = QHBoxLayout()

        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.valueChanged.connect(lambda v: self._update_param(param_name, v))

        layout.addWidget(spinbox)
        group.setLayout(layout)
        self.layout.addWidget(group)

        self.current_params[param_name] = default

    def add_double_spinbox(self, label, param_name, min_val, max_val, default):
        """添加浮点数参数控件"""
        group = QGroupBox(label)
        layout = QHBoxLayout()

        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setSingleStep(0.000001)
        spinbox.setDecimals(6)  # 设置显示的小数位数
        spinbox.valueChanged.connect(lambda v: self._update_param(param_name, v))

        layout.addWidget(spinbox)
        group.setLayout(layout)
        self.layout.addWidget(group)

        self.current_params[param_name] = default

    def add_combobox(self, label, param_name, options):
        """添加下拉选择参数控件"""
        group = QGroupBox(label)
        layout = QHBoxLayout()

        combo = QComboBox()
        combo.addItems(options)
        combo.currentTextChanged.connect(lambda v: self._update_param(param_name, v))

        layout.addWidget(combo)
        group.setLayout(layout)
        self.layout.addWidget(group)

        self.current_params[param_name] = options[0]

    def add_checkbox(self, label, param_name, default):
        """添加复选框参数控件"""
        checkbox = QCheckBox(label)
        checkbox.setChecked(default)
        checkbox.stateChanged.connect(lambda v: self._update_param(param_name, v == Qt.Checked))
        self.layout.addWidget(checkbox)
        self.current_params[param_name] = default

    def _update_param(self, param_name, value):
        """更新参数值"""
        self.current_params[param_name] = value

    def clear_params(self):
        """清除所有参数控件"""
        # while self.layout.count():
        #     item = self.layout.takeAt(0)
        #     widget = item.widget()
        #     if widget:
        #         widget.deleteLater()
        # self.current_params = {}
        # 反向遍历避免索引错位
        for i in reversed(range(self.layout.count())):
            item = self.layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        self.current_params = {}
        # 强制刷新
        self.layout.invalidate()
        self.update()

    def get_params(self):
        """获取当前参数"""
        return self.current_params

    def add_multi_spinbox(self, label, param_name, min_val, max_val):
        """添加多值整数参数输入"""
        group = QGroupBox(f"{label}（多个值用逗号分隔）")
        layout = QHBoxLayout()

        self.line_edit = QLineEdit()
        self.line_edit.setPlaceholderText(f"示例：10,50,100 范围[{min_val}-{max_val}]")
        layout.addWidget(self.line_edit)

        group.setLayout(layout)
        self.layout.addWidget(group)

        self.line_edit.textChanged.connect(
            lambda: self._validate_multi_input(param_name, min_val, max_val)
        )

    def _validate_multi_input(self, param_name, min_val, max_val):
        """验证并解析多值输入"""
        try:
            values = [int(x.strip()) for x in self.sender().text().split(",") if x.strip()]
            valid_values = [v for v in values if min_val <= v <= max_val]
            if valid_values:
                self.current_params[param_name] = valid_values
            else:
                self.current_params[param_name] = [self.parent().default_values[param_name]]
        except:
            self.current_params[param_name] = [self.parent().default_values[param_name]]

class LearningCurveWidget(QWidget):
    """学习曲线显示控件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # 配置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_curve(self, train_sizes, train_scores, test_scores):
        """绘制学习曲线"""
        self.ax.clear()

        self.ax.plot(train_sizes, train_scores, 'o-', color="r", label="训练得分")
        self.ax.plot(train_sizes, test_scores, 'o-', color="g", label="交叉验证得分")

        self.ax.set_xlabel("训练样本数")
        self.ax.set_ylabel("得分")
        self.ax.set_title("学习曲线")
        self.ax.legend(loc="best")

        self.canvas.draw()


class TrainWindow(QDialog):
    """增强版模型训练窗口"""
    model_trained = pyqtSignal(dict)

    def __init__(self, X, y, parent=None, data_fingerprint=None, history=None):
        super().__init__(parent)
        self.X = X
        self.y = y
        self.setWindowTitle("模型训练")
        self.setMinimumSize(900, 700)

        self.trainer = ModelTrainer()
        self.problem_type = self.trainer.determine_problem_type(y)

        # 先初始化 params_widget
        self.params_widget = ModelParamsWidget()

        self._setup_ui()
        self._update_model_combo()  # 现在可以安全调用

        # 修改历史记录存储方式
        self.data_fingerprint = data_fingerprint  # 新增数据指纹
        self.history = history  # 使用主窗口传递的历史记录引用
        self.current_result = None

        # 初始化时加载已有历史
        self._load_existing_history()

    def _setup_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout()
        tab_widget = QTabWidget()

        # 训练选项卡
        train_tab = QWidget()
        train_layout = QVBoxLayout()

        # 模型选择区域
        model_group = QGroupBox("模型配置")
        model_layout = QVBoxLayout()

        # 问题类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("问题类型:"))
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(['自动检测', '分类', '回归', '聚类'])
        self.problem_combo.setCurrentText({
                                              'classification': '分类',
                                              'regression': '回归',
                                              'clustering': '聚类'
                                          }.get(self.problem_type, '自动检测'))
        type_layout.addWidget(self.problem_combo)
        model_layout.addLayout(type_layout)

        # 模型选择
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("选择模型:"))
        self.model_combo = QComboBox()
        model_select_layout.addWidget(self.model_combo)
        model_layout.addLayout(model_select_layout)

        # 参数区域 (使用滚动区域)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.params_widget)  # 使用已经初始化的 params_widget
        model_layout.addWidget(scroll)

        model_group.setLayout(model_layout)
        train_layout.addWidget(model_group)

        # 训练选项区域
        options_group = QGroupBox("训练选项")
        options_layout = QHBoxLayout()

        options_layout.addWidget(QLabel("测试集比例:"))
        self.test_spin = QSpinBox()
        self.test_spin.setRange(10, 50)
        self.test_spin.setValue(20)
        self.test_spin.setSuffix("%")
        options_layout.addWidget(self.test_spin)

        options_layout.addWidget(QLabel("随机种子:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)
        options_layout.addWidget(self.seed_spin)

        self.normalize_check = QCheckBox("标准化数据")
        options_layout.addWidget(self.normalize_check)

        options_group.setLayout(options_layout)
        train_layout.addWidget(options_group)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self._train_model)
        button_layout.addWidget(self.train_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        train_layout.addLayout(button_layout)
        train_tab.setLayout(train_layout)
        tab_widget.addTab(train_tab, "训练")

        # 结果选项卡
        self.results_tab = QWidget()
        results_layout = QVBoxLayout()

        # 在results_layout中添加历史记录选择
        history_layout = QHBoxLayout()
        history_layout.addWidget(QLabel("训练历史:"))

        # 使用QListWidget代替QComboBox
        self.history_list = QListWidget()
        self.history_list.setSelectionMode(QListWidget.SingleSelection)  # 设置单选模式
        self.history_list.itemClicked.connect(self._load_history)  # 点击事件绑定
        history_layout.addWidget(self.history_list)

        # 右侧操作按钮
        history_btn_layout = QVBoxLayout()
        clear_btn = QPushButton("清空历史")
        clear_btn.clicked.connect(self._clear_history)
        history_btn_layout.addWidget(clear_btn)
        history_btn_layout.addStretch()
        history_layout.addLayout(history_btn_layout)

        results_layout.insertLayout(0, history_layout)

        # 指标表格
        self.metrics_table = QTableWidget()
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        results_layout.addWidget(self.metrics_table)

        # 学习曲线
        self.learning_curve = LearningCurveWidget()
        results_layout.addWidget(self.learning_curve)

        self.results_tab.setLayout(results_layout)
        tab_widget.addTab(self.results_tab, "结果")

        # 网格搜索选项卡
        self.grid_search_tab = GridSearchTab(self)
        tab_widget.addTab(self.grid_search_tab, "网格搜索")

        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)

        # 信号连接
        self.problem_combo.currentTextChanged.connect(self._update_model_combo)
        self.model_combo.currentTextChanged.connect(self._update_params)
        self.grid_search_tab.search_completed.connect(self._handle_search_results)

        # 初始化模型下拉框
        self._update_model_combo()

    def _get_model_list(self, problem_type):
        """获取指定问题类型的模型列表"""
        # 复用原有模型列表逻辑
        if problem_type == 'classification':
            return ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN Classification']
        elif problem_type == 'regression':
            return ['Linear Regression', 'Polynomial Regression', 'Ridge Regression',
                    'Lasso Regression', 'Decision Tree', 'Random Forest', 'SVR', 'KNN Regression']
        else:
            return ['K-Means', 'PCA']

    def _handle_search_results(self, results):
        """处理网格搜索结果"""
        # 找出最优模型
        best_result = max(results, key=lambda x: list(x['metrics'].values())[0])
        QMessageBox.information(self, "搜索完成",
            f"最优模型: {best_result['model_name']}\n"
            f"参数: {best_result['params']}\n"
            f"得分: {list(best_result['metrics'].values())[0]:.4f}")

    def _update_model_combo(self):
        """更新模型选择下拉框"""
        current_model = self.model_combo.currentText()
        self.model_combo.clear()

        problem_type = {
            '自动检测': self.problem_type,
            '分类': 'classification',
            '回归': 'regression',
            '聚类': 'clustering'
        }[self.problem_combo.currentText()]

        if problem_type == 'classification':
            self.model_combo.addItems([
                'Logistic Regression',
                'Decision Tree',
                'Random Forest',
                'SVM',
                'KNN Classification'
            ])
        elif problem_type == 'regression':
            self.model_combo.addItems([
                'Linear Regression',
                'Polynomial Regression',
                'Ridge Regression',
                'Lasso Regression',
                'Decision Tree',
                'Random Forest',
                'SVR',
                'KNN Regression'
            ])
        else:  # clustering
            self.model_combo.addItems([
                'K-Means',
                'PCA'
            ])

        # 尝试恢复之前选择的模型
        if current_model in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText(current_model)

        self._update_params()

    def _update_params(self):
        """更新参数控件"""
        problem_type = {
            '自动检测': self.problem_type,
            '分类': 'classification',
            '回归': 'regression',
            '聚类': 'clustering'
        }[self.problem_combo.currentText()]

        model_name = self.model_combo.currentText()
        self.params_widget.update_params(model_name, problem_type)

    # 新增历史记录操作方法
    def _save_to_history(self, result):
        """保存训练结果到历史记录"""
        problem_type = {
            '自动检测': self.problem_type,
            '分类': 'classification',
            '回归': 'regression',
            '聚类': 'clustering'
        }[self.problem_combo.currentText()]

        # 确保保存完整的参数信息
        params = result.get('params', {})
        entry = {
            'id': len(self.history),
            'problem_type': problem_type,
            'model_name': result['model_name'],
            'metrics': result['metrics'],
            'model': result['model'],
            'learning_curve': result.get('learning_curve'),
            'params': params,  # 保存完整参数
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.history.append(entry)

        # 列表项显示格式：模型类型 | 模型名称
        item_text = f"{self._translate_problem_type(entry['problem_type'])} | {entry['model_name']}"
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, entry['id'])

        # 工具提示显示完整参数信息
        param_tooltip = "\n".join([f"{k}: {v}" for k, v in params.items()])
        item.setToolTip(
            f"问题类型: {entry['problem_type']}\n"
            f"训练时间: {entry['timestamp']}\n"
            f"参数:\n{param_tooltip}"
        )
        self.history_list.addItem(item)
        self.history_list.setCurrentItem(item)

    def _translate_problem_type(self, problem_type):
        """翻译问题类型显示"""
        return {
            'classification': '分类',
            'regression': '回归',
            'clustering': '聚类'
        }.get(problem_type, '未知类型')

    def _load_existing_history(self):
        """加载已有历史记录"""
        self.history_list.clear()
        for entry in self.history:
            item = QListWidgetItem(entry['model_name'])
            item.setData(Qt.UserRole, entry['id'])
            self.history_list.addItem(item)

    def _clear_history(self):
        """清空训练历史"""
        self.history.clear()
        self.history_list.clear()  # 正确的控件名称
        self.metrics_table.setRowCount(0)
        self.learning_curve.ax.clear()
        self.learning_curve.canvas.draw()

    def _train_model(self):
        """训练模型"""
        try:
            model_name = self.model_combo.currentText()
            params = self.params_widget.get_params()  # 获取当前参数
            test_size = self.test_spin.value() / 100
            random_state = self.seed_spin.value()
            normalize = self.normalize_check.isChecked()

            # 确定问题类型
            problem_type = {
                '自动检测': self.problem_type,
                '分类': 'classification',
                '回归': 'regression',
                '聚类': 'clustering'
            }[self.problem_combo.currentText()]

            # 训练模型
            result = self.trainer.train_model(
                self.X, self.y,
                model_name=model_name,
                test_size=test_size,
                random_state=random_state,
                normalize=normalize,
                **params
            )

            # 保存训练结果时包含完整的参数信息
            result['model_name'] = model_name
            result['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            result['params'] = params.copy()  # 保存当前参数的副本

            # 训练完成后保存到历史记录
            self._save_to_history(result)
            self._display_results(result)

            # 发送训练完成信号
            self.model_trained.emit({
                'model': result['model_name'],
                'metrics': result['metrics'],
                'problem_type': problem_type
            })

            # 切换到结果选项卡
            self.parent().findChild(QTabWidget).setCurrentIndex(1)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练失败: {str(e)}")

    def _display_results(self, result):
        """显示当前训练结果"""
        problem_type = {
            '自动检测': self.problem_type,
            '分类': 'classification',
            '回归': 'regression',
            '聚类': 'clustering'
        }[self.problem_combo.currentText()]

        # 设置表头
        headers = [
            '问题类型', '模型名称', '训练时间',
            *result['metrics'].keys(),
            '参数摘要'
        ]

        self.metrics_table.setRowCount(1)
        self.metrics_table.setColumnCount(len(headers))
        self.metrics_table.setHorizontalHeaderLabels(headers)

        # 填充数据
        self._set_table_item(0, 0, self._translate_problem_type(problem_type))
        self._set_table_item(0, 1, result['model_name'])
        self._set_table_item(0, 2, result['timestamp'])

        # 填充指标
        for col, (k, v) in enumerate(result['metrics'].items(), start=3):
            self._set_table_item(0, col, v)

        # 参数摘要 - 使用result中的参数而不是当前参数
        params = result.get('params', {})
        params_summary = "; ".join([f"{k}={v}" for k, v in params.items()])
        self._set_table_item(0, len(headers) - 1, params_summary)

        # 显示学习曲线
        if 'learning_curve' in result:
            lc = result['learning_curve']
            self.learning_curve.plot_curve(lc['train_sizes'], lc['train_scores'], lc['test_scores'])

    def _load_history(self, item):
        """加载选中的历史记录"""
        entry_id = item.data(Qt.UserRole)
        entry = next((x for x in self.history if x['id'] == entry_id), None)
        if not entry:
            return

        # 更新指标表格
        self.metrics_table.setRowCount(1)

        # 设置表头
        headers = [
            '问题类型', '模型名称', '训练时间',
            *entry['metrics'].keys(),
            '参数摘要'
        ]
        self.metrics_table.setColumnCount(len(headers))
        self.metrics_table.setHorizontalHeaderLabels(headers)

        # 填充数据
        self._set_table_item(0, 0, self._translate_problem_type(entry['problem_type']))
        self._set_table_item(0, 1, entry['model_name'])
        self._set_table_item(0, 2, entry['timestamp'])

        # 填充指标
        for col, (_, v) in enumerate(entry['metrics'].items(), start=3):
            self._set_table_item(0, col, v)

        # 参数摘要 - 使用历史记录中的参数
        params = entry.get('params', {})
        params_summary = "; ".join([f"{k}={v}" for k, v in params.items()])
        self._set_table_item(0, len(headers) - 1, params_summary)

        # 显示学习曲线
        if entry.get('learning_curve'):
            self.learning_curve.plot_curve(
                entry['learning_curve']['train_sizes'],
                entry['learning_curve']['train_scores'],
                entry['learning_curve']['test_scores']
            )
    def _set_table_item(self, row, col, value):
        """辅助方法：设置表格项"""
        if isinstance(value, (list, np.ndarray)):
            text = ', '.join([f'{v:.3f}' for v in value])
        elif isinstance(value, float):
            text = f'{value:.4f}'
        else:
            text = str(value)
        self.metrics_table.setItem(row, col, QTableWidgetItem(text))


class GridSearchTab(QWidget):
    """网格搜索选项卡"""
    search_completed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.results = []
        self._setup_ui()
        # 初始化时设置正确的问题类型
        self.problem_combo.setCurrentText({
            'classification': '分类',
            'regression': '回归',
            'clustering': '聚类'
        }.get(self.parent_window.problem_type, '自动检测'))

    def _setup_ui(self):
        # 使用垂直分割器实现可调整比例
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(5)  # 设置分隔条宽度
        splitter.setStyleSheet("QSplitter::handle { background: #ccc; }")

        # 配置区域
        config_group = QGroupBox("搜索配置")
        config_layout = QVBoxLayout()

        # 问题类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("问题类型:"))
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(['自动检测', '分类', '回归', '聚类'])
        type_layout.addWidget(self.problem_combo)
        config_layout.addLayout(type_layout)

        # 模型选择
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("选择模型:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        config_layout.addLayout(model_layout)

        # 参数配置区域
        self.params = ModelParams()

        # 参数范围设置区域
        self.param_range_group = QGroupBox("参数范围设置")
        param_range_layout = QVBoxLayout()

        # 添加参数范围控件
        self.param_range_controls = {}
        self._setup_param_range_controls(param_range_layout)

        self.param_range_group.setLayout(param_range_layout)
        config_layout.addWidget(self.param_range_group)

        # 公共参数
        common_layout = QHBoxLayout()
        common_layout.addWidget(QLabel("随机种子:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)
        common_layout.addWidget(self.seed_spin)

        self.normalize_check = QCheckBox("标准化数据")
        common_layout.addWidget(self.normalize_check)
        config_layout.addLayout(common_layout)

        config_group.setLayout(config_layout)

        # 操作按钮
        btn_layout = QHBoxLayout()
        self.search_btn = QPushButton("开始搜索")
        self.search_btn.clicked.connect(self._run_grid_search)
        btn_layout.addWidget(self.search_btn)


        # 结果展示
        result_group = QGroupBox("搜索结果")
        result_layout = QVBoxLayout()

        # 可视化选择
        self.viz_combo = QComboBox()
        self.viz_combo.addItems(["表格视图", "箱线图比较"])
        result_layout.addWidget(self.viz_combo)

        # 结果表格
        self.result_table = QTableWidget()
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        result_layout.addWidget(self.result_table)

        # 图表区域
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(200)  # 防止高度为0
        result_layout.addWidget(self.canvas)

        # 按钮
        result_layout.addLayout(btn_layout) # 添加按钮到布局

        result_group.setLayout(result_layout)

        # 将两个区域添加到分割器
        splitter.addWidget(config_group)
        splitter.addWidget(result_group)
        # 设置初始比例（40%:60%）
        splitter.setSizes([int(self.height()*0.4), int(self.height()*0.6)])

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
        # 信号连接
        self.problem_combo.currentTextChanged.connect(self._update_model_combo)
        self.model_combo.currentTextChanged.connect(self._update_params_ui)

    def _current_problem_type(self):
        """获取当前选择的问题类型（处理自动检测）"""
        selected_type = self.problem_combo.currentText()
        if selected_type == '自动检测':
            return self.parent_window.problem_type
        return {
            '分类': 'classification',
            '回归': 'regression',
            '聚类': 'clustering'
        }[selected_type]
    def _setup_param_range_controls(self, layout):
        """设置参数范围控件"""
        # 清空现有控件
        for i in reversed(range(layout.count())):
            item = layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self.param_range_controls = {}

    def _update_model_combo(self):
        """更新模型选择下拉框"""
        current_model = self.model_combo.currentText()
        self.model_combo.clear()

        problem_type = self._current_problem_type()  # 使用修正后的方法

        self.model_combo.addItems(self.parent_window._get_model_list(problem_type))

        # 尝试恢复之前选择的模型
        if current_model in [self.model_combo.itemText(i) for i in range(self.model_combo.count())]:
            self.model_combo.setCurrentText(current_model)

        self._update_params_ui()

    def _update_params_ui(self):
        """更新参数UI"""
        problem_type = self._current_problem_type()  # 使用修正后的方法

        model_name = self.model_combo.currentText()
        self.params.update_params(model_name, problem_type)

        # 更新参数范围控件
        self._update_param_range_controls()

    def _clear_all_controls(self):
        """递归清除参数范围组内的所有控件"""
        def recursive_clear(layout):
            while layout.count() > 0:
                item = layout.takeAt(0)
                widget = item.widget()
                if widget:
                    # 如果控件有子布局，先清理子布局
                    if widget.layout():
                        recursive_clear(widget.layout())
                    # 断开信号并删除控件
                    widget.setParent(None)
                    widget.deleteLater()
                else:
                    sub_layout = item.layout()
                    if sub_layout:
                        recursive_clear(sub_layout)

        layout = self.param_range_group.layout()
        recursive_clear(layout)
    def _update_param_range_controls(self):
        """更新参数范围控件"""
        # 清空现有控件
        layout = self.param_range_group.layout()
        # while layout.count() > 0:
        #     item = layout.takeAt(0)
        #     widget = item.widget()
        #     if widget:
        #         widget.deleteLater()
        self._clear_all_controls()  # 替换原有的清理逻辑

        self.param_range_controls = {}
        params = self.params.get_params()

        # 为每个参数添加设置控件
        for param_name, default_value in params.items():
            param_info = self.params.get_param_info(param_name)
            if not param_info:
                continue

            if param_info.get('type') == 'bool':
                # 布尔类型参数 - 添加多选复选框
                group = QGroupBox(f"{param_name}选项")
                param_layout = QVBoxLayout()

                options_group = QGroupBox("选择要测试的值（可多选）")
                options_layout = QVBoxLayout()

                self.param_range_controls[param_name] = {
                    'type': 'bool',
                    'checkboxes': {
                        True: QCheckBox("True"),
                        False: QCheckBox("False")
                    }
                }

                # 默认全选
                for cb in self.param_range_controls[param_name]['checkboxes'].values():
                    cb.setChecked(True)
                    options_layout.addWidget(cb)

                options_group.setLayout(options_layout)
                param_layout.addWidget(options_group)
                group.setLayout(param_layout)
                layout.addWidget(group)

            elif param_info.get('type') == 'options':
                # 字符串型参数 - 添加多选控件
                group = QGroupBox(f"{param_name}选项")
                param_layout = QVBoxLayout()

                options = param_info['options']
                if options:
                    options_group = QGroupBox("选择要测试的值")
                    options_layout = QVBoxLayout()

                    self.param_range_controls[param_name] = {
                        'type': 'options',
                        'checkboxes': []
                    }

                    for option in options:
                        cb = QCheckBox(option)
                        cb.setChecked(True)
                        options_layout.addWidget(cb)
                        self.param_range_controls[param_name]['checkboxes'].append(cb)

                    options_group.setLayout(options_layout)
                    param_layout.addWidget(options_group)

                group.setLayout(param_layout)
                layout.addWidget(group)
            else:
                # 数值型参数 - 添加完整范围设置组
                group = QGroupBox(f"{param_name}范围设置")
                param_layout = QFormLayout()

                # 判断是整数还是浮点数
                is_float = isinstance(default_value, (float, np.floating))

                # 最小值
                min_spin = QDoubleSpinBox() if is_float else QSpinBox()
                min_spin.setValue(float(param_info.get('min', default_value)))
                min_spin.setMinimum(-99999)
                min_spin.setDecimals(6)  # 设置显示的小数位数
                param_layout.addRow("最小值:", min_spin)

                # 最大值
                max_spin = QDoubleSpinBox() if is_float else QSpinBox()
                max_spin.setValue(float(param_info.get('max', default_value)))
                max_spin.setMinimum(-99999)
                max_spin.setDecimals(6)  # 设置显示的小数位数
                param_layout.addRow("最大值:", max_spin)

                # 步长
                step_spin = QDoubleSpinBox()
                step_spin.setRange(0.000001, 100)
                step_spin.setValue(1.0 if not is_float else 0.000001)
                step_spin.setSingleStep(1.0 if not is_float else 0.000001)
                step_spin.setDecimals(6)  # 设置显示的小数位数
                param_layout.addRow("步长:", step_spin)

                group.setLayout(param_layout)
                layout.addWidget(group)

                self.param_range_controls[param_name] = {
                    'type': 'range',
                    'min': min_spin,
                    'max': max_spin,
                    'step': step_spin
                }

    def _get_param_ranges(self):
        """获取参数范围/选项设置"""
        param_ranges = {}

        for param_name, control in self.param_range_controls.items():
            if control['type'] == 'range':
                # 数值型参数
                min_val = control['min'].value()
                max_val = control['max'].value()
                step = control['step'].value()

                if min_val > max_val:
                    min_val, max_val = max_val, min_val

                if isinstance(control['min'], QDoubleSpinBox):
                    param_ranges[param_name] = np.arange(min_val, max_val + step / 2, step)
                else:
                    param_ranges[param_name] = np.arange(
                        int(min_val), int(max_val) + 1, int(step))

            elif control['type'] == 'options':
                # 字符串型参数
                selected_options = [cb.text() for cb in control['checkboxes'] if cb.isChecked()]
                if selected_options:
                    param_ranges[param_name] = selected_options

            elif control['type'] == 'bool':
                # 收集选中的布尔值
                selected = []
                for value, cb in control['checkboxes'].items():
                    if cb.isChecked():
                        selected.append(value)
                param_ranges[param_name] = selected

        return param_ranges
    def _run_grid_search(self):
        """执行网格搜索"""
        try:
            self.search_btn.setEnabled(False)
            QApplication.setOverrideCursor(Qt.WaitCursor)

            # 创建进度对话框
            progress = QProgressDialog("正在执行网格搜索...", "取消", 0, 100, self)
            progress.setWindowTitle("进度")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            model_name = self.model_combo.currentText()
            base_params = self.params.get_params()
            param_ranges = self._get_param_ranges()

            # 生成参数网格
            param_grid = []
            for param_name, values in param_ranges.items():
                for value in values:
                    params = base_params.copy()
                    params[param_name] = value
                    param_grid.append(params)

            # 执行搜索
            self.results = []
            total = len(param_grid)
            for i, params in enumerate(param_grid, 1):
                if progress.wasCanceled():
                    break

                progress.setValue(int(i / total * 100))
                progress.setLabelText(f"正在训练 {model_name}... ({i}/{total})")
                QApplication.processEvents()

                result = self._train_single_model(model_name, params)
                self.results.append(result)

            if not progress.wasCanceled():
                self._display_results()
                self._show_best_result()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"网格搜索失败: {str(e)}")
        finally:
            self.search_btn.setEnabled(True)
            QApplication.restoreOverrideCursor()

    def _get_scoring_metric(self):
        """根据问题类型获取评分指标"""
        problem_type = self._current_problem_type()
        return 'accuracy' if problem_type == 'classification' else 'r2'

    def _train_single_model(self, model_name, params):
        """训练单个模型"""
        # 复用主窗口的训练逻辑
        result = self.parent_window.trainer.train_model(
            self.parent_window.X,
            self.parent_window.y,
            model_name=model_name,
            test_size=0.2,
            random_state=self.seed_spin.value(),
            normalize=self.normalize_check.isChecked(),
            **params
        )
        scoring = self._get_scoring_metric()  # 新增方法获取评分指标
        # 获取交叉验证分数
        cv_scores = self.parent_window.trainer.get_cross_val_scores(
            self.parent_window.X,
            self.parent_window.y,
            model_name=model_name,
            params=params,
            scoring=scoring
        )
        return {
            'model_name': model_name,
            'params': params,
            'metrics': result['metrics'],
            'model': result['model'],
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores)
        }

    def _display_results(self):
        self.figure.clear()

        # 正确提取模型名称和参数
        data = []
        for r in self.results:
            entry = {
                '模型名称': r['model_name'],  # 使用明确记录的模型名称
                **r['metrics'],
                '参数': "; ".join([f"{k}={v}" for k, v in r['params'].items()])
            }
            data.append(entry)

        df = pd.DataFrame(data)  # 使用修正后的数据结构
        print(self.viz_combo.currentText())
        if self.viz_combo.currentText() == "表格视图":
            self._show_table(df)
        else:
            self._show_boxplot()
    def _show_best_result(self):
        """显示最优结果"""
        if not self.results:
            return

        metric = list(self.results[0]['metrics'].keys())[0]  # 取第一个指标
        best = max(self.results, key=lambda x: x['metrics'][metric])

        msg = f"""🏆 最优模型配置：
        • 模型类型：{best['model']}
        • 参数组合：{best['params']}
        • {metric}：{best['metrics'][metric]:.4f}
        """

        QMessageBox.information(self, "搜索完成", msg)

    def _show_table(self, df):
        self.result_table.setRowCount(len(df))
        self.result_table.setColumnCount(len(df.columns))
        self.result_table.setHorizontalHeaderLabels(df.columns)

        for i, row in df.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                if j == 0:  # 模型列特殊标记
                    item.setBackground(QColor(240, 240, 240))
                self.result_table.setItem(i, j, item)

        self.result_table.setVisible(True)
        self.canvas.setVisible(False)

    def _show_boxplot(self):
        """显示箱线图比较"""
        ax = self.figure.add_subplot(111)
        print(self.results)
        # 准备箱线图数据
        data = [r['cv_scores'] for r in self.results]
        labels = [self._format_params(r['params']) for r in self.results]

        # 绘制箱线图
        ax.boxplot(data, labels=labels)
        ax.set_title(f"{'Accuracy' if self._get_scoring_metric() == 'accuracy' else 'R2 Score'} 分布比较")
        ax.set_ylabel("Score")
        plt.xticks(rotation=45, ha='right')

        self.canvas.draw()
        self.result_table.setVisible(False)
        self.canvas.setVisible(True)

    def _format_params(self, params):
        """格式化参数显示为简短字符串"""
        return ", ".join([f"{k[:3]}={v}" for k, v in params.items()])

class ModelParams:
    """纯参数逻辑类，不包含UI控件"""

    def __init__(self):
        self.current_params = {}

    def update_params(self, model_name, problem_type):
        """根据模型更新参数"""
        self.clear_params()

        if problem_type == 'regression':
            if model_name == 'Linear Regression':
                self._add_param('fit_intercept', True)
            elif model_name == 'Polynomial Regression':
                self._add_param('degree', 2, min_val=1, max_val=10)
            elif model_name == 'Ridge Regression':
                self._add_param('alpha', 1.0, min_val=0.01, max_val=100.0)
            elif model_name == 'Lasso Regression':
                self._add_param('alpha', 1.0, min_val=0.01, max_val=100.0)
            elif model_name == 'Decision Tree':
                self._add_param('max_depth', 5, min_val=1, max_val=50)
                self._add_param('min_samples_split', 2, min_val=2, max_val=20)
            elif model_name == 'Random Forest':
                self._add_param('n_estimators', 100, min_val=10, max_val=500)
                self._add_param('max_depth', 5, min_val=1, max_val=50)
            elif model_name == 'SVR':
                self._add_param('kernel', 'rbf', options=['linear', 'poly', 'rbf', 'sigmoid'])
                self._add_param('C', 1.0, min_val=1, max_val=10)
            elif model_name == 'KNN Regression':
                self._add_param('n_neighbors', 5, min_val=1, max_val=20)

        elif problem_type == 'classification':
            if model_name == 'Logistic Regression':
                self._add_param('C', 1.0, min_val=0.01, max_val=100.0)
                self._add_param('penalty', 'l2', options=['l1', 'l2', 'elasticnet', 'none'])
            elif model_name == 'Decision Tree':
                self._add_param('max_depth', 5, min_val=1, max_val=50)
                self._add_param('min_samples_split', 2, min_val=2, max_val=20)
            elif model_name == 'Random Forest':
                self._add_param('n_estimators', 100, min_val=10, max_val=500)
                self._add_param('max_depth', 5, min_val=1, max_val=50)
            elif model_name == 'SVM':
                self._add_param('kernel', 'rbf', options=['linear', 'poly', 'rbf', 'sigmoid'])
                self._add_param('C', 1.0, min_val=0.1, max_val=100.0)
            elif model_name == 'KNN Classification':
                self._add_param('n_neighbors', 5, min_val=1, max_val=20)

        elif problem_type == 'clustering':
            if model_name == 'K-Means':
                self._add_param('n_clusters', 5, min_val=2, max_val=20)
            elif model_name == 'PCA':
                self._add_param('n_components', 2, min_val=1, max_val=20)

    def _add_param(self, param_name, default_value, min_val=None, max_val=None, options=None):
        """添加参数"""
        self.current_params[param_name] = default_value

        # 保存额外信息用于网格搜索
        param_info = {
            'default': default_value,
            'type': 'bool' if isinstance(default_value, bool) else
            'numeric' if (min_val is not None and max_val is not None) else
            'options' if options is not None else 'other'
        }

        if min_val is not None:
            param_info['min'] = min_val
        if max_val is not None:
            param_info['max'] = max_val
        if options is not None:
            param_info['options'] = options

        setattr(self, f'_{param_name}_info', param_info)

    def clear_params(self):
        """清除所有参数"""
        self.current_params = {}

    def get_params(self):
        """获取当前参数"""
        return self.current_params.copy()

    def get_param_info(self, param_name):
        """获取参数的额外信息（用于网格搜索）"""
        return getattr(self, f'_{param_name}_info', None)