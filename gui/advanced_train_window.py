from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QComboBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSpinBox,
                             QCheckBox, QMessageBox, QLineEdit, QFormLayout,
                             QWidget, QScrollArea, QDoubleSpinBox, QSplitter,
                             QProgressDialog)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import pandas as pd
from datetime import datetime

from skimage.metrics import mean_squared_error
from sklearn.ensemble import (BaggingClassifier, BaggingRegressor,
                              AdaBoostClassifier, AdaBoostRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler

from core.model_trainer import ModelTrainer

from PyQt5.QtWidgets import QApplication
import sys


class AdvancedTrainWindow(QDialog):
    """高级模型训练窗口 - 专门用于集成学习算法"""
    model_trained = pyqtSignal(dict)

    def __init__(self, X, y, parent=None, data_fingerprint=None, history=None):
        super().__init__(parent)
        self.X = X
        self.y = y
        self.setWindowTitle("高级模型训练 - 集成学习")
        self.setMinimumSize(900, 700)

        self.trainer = ModelTrainer()
        self.problem_type = self.trainer.determine_problem_type(y)

        # 初始化集成学习模型
        self.ensemble_models = {
            'classification': {
                'Bagging': BaggingClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'Gradient Boosting': GradientBoostingClassifier()
            },
            'regression': {
                'Bagging': BaggingRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor()
            }
        }

        # 历史记录和数据指纹
        self.data_fingerprint = data_fingerprint
        self.history = history if history is not None else []
        self.current_result = None

        self._setup_ui()

    def _setup_ui(self):
        """设置UI界面"""
        main_layout = QVBoxLayout()
        splitter = QSplitter(Qt.Vertical)

        # 上部配置区域
        config_widget = QWidget()
        config_layout = QVBoxLayout()

        # 模型选择区域
        model_group = QGroupBox("集成模型配置")
        model_layout = QFormLayout()

        # 模型类型选择
        self.model_combo = QComboBox()
        model_names = list(self.ensemble_models[self.problem_type].keys())
        self.model_combo.addItems(model_names)
        model_layout.addRow("选择集成算法:", self.model_combo)

        # 基础学习器选择
        self.base_estimator_combo = QComboBox()
        self._update_base_estimators()
        model_layout.addRow("基础学习器:", self.base_estimator_combo)

        # 参数配置区域
        self.params_group = QGroupBox("参数配置")
        params_layout = QFormLayout()

        # 公共参数
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(100)
        params_layout.addRow("学习器数量:", self.n_estimators_spin)

        self.learning_rate_spin = QDoubleSpinBox()
        self.learning_rate_spin.setRange(0.01, 1.0)
        self.learning_rate_spin.setValue(0.1)
        self.learning_rate_spin.setSingleStep(0.01)
        params_layout.addRow("学习率:", self.learning_rate_spin)

        # 算法特定参数
        self.specific_params_group = QGroupBox("算法特定参数")
        self.specific_params_layout = QFormLayout()
        self._update_specific_params()
        self.specific_params_group.setLayout(self.specific_params_layout)

        params_layout.addRow(self.specific_params_group)
        self.params_group.setLayout(params_layout)

        model_layout.addRow(self.params_group)
        model_group.setLayout(model_layout)
        config_layout.addWidget(model_group)

        # 训练选项
        options_group = QGroupBox("训练选项")
        options_layout = QHBoxLayout()

        self.test_size_spin = QSpinBox()
        self.test_size_spin.setRange(10, 50)
        self.test_size_spin.setValue(20)
        self.test_size_spin.setSuffix("%")
        options_layout.addWidget(QLabel("测试集比例:"))
        options_layout.addWidget(self.test_size_spin)

        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 9999)
        self.random_state_spin.setValue(42)
        options_layout.addWidget(QLabel("随机种子:"))
        options_layout.addWidget(self.random_state_spin)

        self.normalize_check = QCheckBox("标准化数据")
        options_layout.addWidget(self.normalize_check)

        options_group.setLayout(options_layout)
        config_layout.addWidget(options_group)

        # 训练按钮
        self.train_btn = QPushButton("开始训练")
        self.train_btn.clicked.connect(self._train_model)
        config_layout.addWidget(self.train_btn)

        config_widget.setLayout(config_layout)
        splitter.addWidget(config_widget)

        # 下部结果区域
        result_widget = QWidget()
        result_layout = QVBoxLayout()

        # 历史记录
        history_group = QGroupBox("训练历史")
        history_layout = QVBoxLayout()

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(6)
        self.history_table.setHorizontalHeaderLabels(
            ["时间", "算法", "基础学习器", "学习器数量", "评分", "参数"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)

        history_layout.addWidget(self.history_table)
        history_group.setLayout(history_layout)
        result_layout.addWidget(history_group)

        # 指标表格
        metrics_group = QGroupBox("训练指标")
        metrics_layout = QVBoxLayout()

        self.metrics_table = QTableWidget()
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        metrics_layout.addWidget(self.metrics_table)
        metrics_group.setLayout(metrics_layout)
        result_layout.addWidget(metrics_group)

        result_widget.setLayout(result_layout)
        splitter.addWidget(result_widget)

        # 设置初始比例
        splitter.setSizes([300, 400])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        # 信号连接
        self.model_combo.currentTextChanged.connect(self._update_specific_params)
        self.base_estimator_combo.currentTextChanged.connect(self._update_specific_params)

        # 加载历史记录
        self._load_history()

    def _update_base_estimators(self):
        """更新基础学习器选项"""
        self.base_estimator_combo.clear()

        algorithm = self.model_combo.currentText()

        if self.problem_type == 'classification':
            estimators = ['Decision Tree', 'Logistic Regression']
            if algorithm != 'AdaBoost':  # KNN和SVM不支持sample_weight
                estimators.extend(['SVM', 'KNN Classification'])
            self.base_estimator_combo.addItems(estimators)
        else:  # regression
            estimators = ['Decision Tree', 'Linear Regression']
            if algorithm != 'AdaBoost':
                estimators.extend(['SVR', 'KNN Regression'])
            self.base_estimator_combo.addItems(estimators)

    def _update_specific_params(self):
        """更新算法特定参数"""
        # 清空现有参数
        for i in reversed(range(self.specific_params_layout.count())):
            item = self.specific_params_layout.takeAt(i)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        algorithm = self.model_combo.currentText()
        base_estimator = self.base_estimator_combo.currentText()

        # 根据算法类型显示/隐藏学习率控件
        if algorithm in ['AdaBoost', 'Gradient Boosting']:
            self.learning_rate_spin.setEnabled(True)
        else:
            self.learning_rate_spin.setEnabled(False)

        # 添加算法特定参数
        if algorithm == 'Bagging':
            self.max_samples_spin = QDoubleSpinBox()
            self.max_samples_spin.setRange(0.1, 1.0)
            self.max_samples_spin.setValue(1.0)
            self.max_samples_spin.setSingleStep(0.1)
            self.specific_params_layout.addRow("最大样本比例:", self.max_samples_spin)

            self.max_features_spin = QDoubleSpinBox()
            self.max_features_spin.setRange(0.1, 1.0)
            self.max_features_spin.setValue(1.0)
            self.max_features_spin.setSingleStep(0.1)
            self.specific_params_layout.addRow("最大特征比例:", self.max_features_spin)

        elif algorithm == 'AdaBoost':
            if base_estimator == 'Decision Tree':
                self.max_depth_spin = QSpinBox()
                self.max_depth_spin.setRange(1, 20)
                self.max_depth_spin.setValue(3)
                self.specific_params_layout.addRow("决策树最大深度:", self.max_depth_spin)

        elif algorithm == 'Gradient Boosting':
            self.subsample_spin = QDoubleSpinBox()
            self.subsample_spin.setRange(0.1, 1.0)
            self.subsample_spin.setValue(1.0)
            self.subsample_spin.setSingleStep(0.1)
            self.specific_params_layout.addRow("子采样比例:", self.subsample_spin)

            self.max_depth_spin = QSpinBox()
            self.max_depth_spin.setRange(1, 20)
            self.max_depth_spin.setValue(3)
            self.specific_params_layout.addRow("最大深度:", self.max_depth_spin)

            self.min_samples_split_spin = QSpinBox()
            self.min_samples_split_spin.setRange(2, 20)
            self.min_samples_split_spin.setValue(2)
            self.specific_params_layout.addRow("最小分割样本数:", self.min_samples_split_spin)
    def _get_model_params(self):
        """获取当前参数配置"""
        algorithm = self.model_combo.currentText()
        params = {
            'n_estimators': self.n_estimators_spin.value(),
            'random_state': self.random_state_spin.value()
        }

        # 只有AdaBoost和Gradient Boosting需要learning_rate
        if algorithm in ['AdaBoost', 'Gradient Boosting']:
            params['learning_rate'] = self.learning_rate_spin.value()

        if algorithm == 'Bagging':
            params.update({
                'max_samples': self.max_samples_spin.value(),
                'max_features': self.max_features_spin.value()
            })
        elif algorithm == 'AdaBoost':
            base_estimator = self.base_estimator_combo.currentText()
            if base_estimator == 'Decision Tree':
                params.update({
                    'base_estimator__max_depth': self.max_depth_spin.value()
                })
        elif algorithm == 'Gradient Boosting':
            params.update({
                'subsample': self.subsample_spin.value(),
                'max_depth': self.max_depth_spin.value(),
                'min_samples_split': self.min_samples_split_spin.value()
            })

        return params
    def _train_model(self):
        """训练集成模型"""
        try:
            # 创建进度对话框
            progress = QProgressDialog("正在训练模型...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # 获取参数
            algorithm = self.model_combo.currentText()
            base_estimator_name = self.base_estimator_combo.currentText()
            params = self._get_model_params()
            test_size = self.test_size_spin.value() / 100
            random_state = self.random_state_spin.value()
            normalize = self.normalize_check.isChecked()

            params['random_state'] = random_state
            # 设置基础学习器
            if algorithm in ['Bagging', 'AdaBoost']:
                # 直接从ModelTrainer获取基础学习器
                base_estimator = self.trainer.models[self.problem_type][base_estimator_name]
                params['base_estimator'] = base_estimator

            # 获取模型 - 直接从self.ensemble_models获取
            model = self.ensemble_models[self.problem_type][algorithm]
            model.set_params(**params)

            # 更新进度
            progress.setValue(20)
            QApplication.processEvents()

            # 划分数据集
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=test_size, random_state=random_state)

            print("训练集类别分布:", np.unique(y_train, return_counts=True))
            print("测试集类别分布:", np.unique(y_test, return_counts=True))

            # 数据标准化
            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                self.scaler = scaler

            # 训练模型
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 评估模型
            metrics = self._evaluate_model(y_test, y_pred)

            # 获取学习曲线
            lc = self._get_learning_curve(model, self.X, self.y)

            # 更新进度
            progress.setValue(80)
            QApplication.processEvents()

            # 保存结果
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            score = list(metrics.values())[-1]  # 取第一个指标作为代表分数

            history_entry = {
                'id': len(self.history),
                'model_name': algorithm + " (" + base_estimator_name + ")",
                'timestamp': timestamp,
                'algorithm': algorithm,
                'base_estimator': base_estimator_name,
                'n_estimators': params['n_estimators'],
                'score': score,
                'params': params,
                'model': model,
                'metrics': metrics,
                'learning_curve': lc,
                'problem_type': self.problem_type
            }

            self.history.append(history_entry)
            self._add_history_row(history_entry)
            self._show_metrics(metrics)

            # 发送训练完成信号
            self.model_trained.emit({
                'model': model,
                'metrics': metrics,
                'problem_type': self.problem_type,
                'model_name': f"{algorithm} ({base_estimator_name})",
                'timestamp': timestamp
            })

            progress.setValue(100)
            # QMessageBox.information(self, "成功", "模型训练完成!")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练失败: {str(e)}")
        finally:
            progress.close()

    def _evaluate_model(self, y_true, y_pred):
        """评估模型性能"""
        metrics = {}
        if self.problem_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
        else:  # regression
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        return metrics

    def _get_learning_curve(self, model, X, y, cv=5):
        """获取学习曲线数据"""
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 5))

        return {
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1)
        }
    def _load_history(self):
        """加载历史记录"""
        self.history_table.setRowCount(0)
        for entry in self.history:
            self._add_history_row(entry)

    def _add_history_row(self, entry):
        if entry.get('algorithm') is None:
            return
        """添加一行历史记录"""
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)

        self.history_table.setItem(row, 0, QTableWidgetItem(entry['timestamp']))
        self.history_table.setItem(row, 1, QTableWidgetItem(entry['algorithm']))
        self.history_table.setItem(row, 2, QTableWidgetItem(entry.get('base_estimator', 'N/A')))
        self.history_table.setItem(row, 3, QTableWidgetItem(str(entry['n_estimators'])))
        self.history_table.setItem(row, 4, QTableWidgetItem(f"{entry['score']:.4f}"))

        # 简化参数显示
        params_text = ", ".join([f"{k.split('__')[-1]}={v}" for k, v in entry['params'].items()
                                 if k not in ['n_estimators', 'random_state']])
        self.history_table.setItem(row, 5, QTableWidgetItem(params_text))

    def _show_metrics(self, metrics):
        """显示评估指标"""
        self.metrics_table.setRowCount(len(metrics))
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "值"])

        for i, (name, value) in enumerate(metrics.items()):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{value:.4f}"))

