import pandas as pd
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QComboBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QFileDialog,
                             QMessageBox, QFormLayout, QWidget, QSplitter)
from PyQt5.QtCore import Qt
import numpy as np
from datetime import datetime

from sklearn.metrics import accuracy_score, r2_score, silhouette_score

from core.data_loader import DataLoader


class PredictWindow(QDialog):
    """增强版模型预测窗口"""

    def __init__(self, X, target, history, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型预测")
        self.setMinimumSize(1200, 800)

        self.X = X  # 训练数据特征
        self.target = target
        self.history = history if isinstance(history, list) else []
        self.current_model = None
        self.predict_data = None

        self.true_labels = None  # 真实标签（如果有）
        self.predictions = []
        self.scaler = None

        self._setup_ui()

    def _setup_ui(self):
        """初始化UI"""
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Horizontal)

        # 左侧配置区域
        left_panel = QWidget()
        left_layout = QVBoxLayout()

        # 模型选择
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        self.model_combo = QComboBox()
        self._populate_model_combo()
        model_layout.addRow("选择模型:", self.model_combo)
        model_group.setLayout(model_layout)
        left_layout.addWidget(model_group)

        # 数据上传
        data_group = QGroupBox("数据配置")
        data_layout = QVBoxLayout()

        # 特征数据上传
        self.feature_btn = QPushButton("上传特征数据")
        self.feature_btn.clicked.connect(lambda: self._upload_data(is_features=True))
        data_layout.addWidget(self.feature_btn)

        # 标签数据上传（可选）
        self.label_btn = QPushButton("上传真实标签（可选）")
        self.label_btn.clicked.connect(lambda: self._upload_data(is_features=False))
        data_layout.addWidget(self.label_btn)

        self.data_info = QLabel("等待数据上传...")
        self.data_info.setAlignment(Qt.AlignCenter)
        data_layout.addWidget(self.data_info)

        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)

        # 预测按钮
        self.predict_btn = QPushButton("执行预测")
        self.predict_btn.clicked.connect(self._predict)
        self.predict_btn.setEnabled(False)
        left_layout.addWidget(self.predict_btn)

        left_panel.setLayout(left_layout)

        # 右侧结果区域
        right_panel = QWidget()
        right_layout = QVBoxLayout()

        # 预测结果
        result_group = QGroupBox("预测结果")
        result_layout = QVBoxLayout()

        self.result_table = QTableWidget()
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.result_table)

        # 得分显示
        self.score_label = QLabel("评估得分：等待预测...")
        result_layout.addWidget(self.score_label)

        # 导出按钮
        self.export_btn = QPushButton("导出结果")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        result_layout.addWidget(self.export_btn)

        result_group.setLayout(result_layout)
        right_layout.addWidget(result_group)

        # 预测记录
        history_group = QGroupBox("预测记录")
        history_layout = QVBoxLayout()

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["时间", "模型", "样本数", "得分类型", "得分"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table)

        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group)

        right_panel.setLayout(right_layout)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _populate_model_combo(self):
        """填充模型下拉框"""
        self.model_combo.clear()

        if not self.history:
            QMessageBox.warning(self, "警告", "没有可用的训练模型")
            return

        for entry in self.history:
            if isinstance(entry, dict) and 'model_name' in entry and 'timestamp' in entry:
                item_text = f"{entry['model_name']} ({entry['timestamp']})"
                self.model_combo.addItem(item_text, entry)

    def _upload_data(self, is_features=True):
        """上传数据"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开数据文件", "",
            "数据文件 (*.csv *.xls *.xlsx *.txt);;所有文件 (*.*)"
        )

        if file_path:
            try:
                dataloader = DataLoader()
                data = dataloader.load_file(file_path)
                if data is None:
                    QMessageBox.warning(self, "格式错误", "不支持的文件格式")
                    return
                if is_features:
                    # 检查特征列
                    if not set(data.columns).issubset(set(self.X.columns) - {self.target}):
                        missing_cols = set(self.X.columns) - set(data.columns)
                        QMessageBox.warning(
                            self, "列不匹配",
                            f"特征数据缺少以下列: {', '.join(missing_cols)}\n"
                            "请确保数据包含所有训练时使用的特征列"
                        )
                        return
                    self.predict_data = data[self.X.columns]
                    self.data_info.setText(
                        f"已加载特征数据: {len(data)}行 × {len(data.columns)}列"
                    )
                else:
                    # 假设标签数据只有一列
                    if len(data.columns) != 1:
                        QMessageBox.warning(self, "格式错误", "标签数据应包含单列")
                        return
                    self.true_labels = data.iloc[:, 0]
                    self.data_info.setText(
                        f"{self.data_info.text()}\n已加载标签数据: {len(data)}样本"
                    )

                if self.predict_data is not None:
                    self.predict_btn.setEnabled(True)

            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载数据失败: {str(e)}")

    def _predict(self):
        """执行预测"""
        try:
            model = self.model_combo.currentData()['model']
            problem_type = self.model_combo.currentData()['problem_type']

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 执行预测
            if hasattr(model, 'predict'):
                predictions = model.predict(self.predict_data)
                result_df = self._format_prediction_result(predictions, model)
            else:
                result_df = self._handle_unsupervised(model)

            # 显示结果
            self._display_results(result_df)

            # 计算得分（如果有标签）
            score_info = self._calculate_scores(predictions, model, problem_type)

            # 记录预测信息
            self._add_prediction_record(
                timestamp=timestamp,
                model_name=self.model_combo.currentText(),
                sample_count=len(self.predict_data),
                score_info=score_info
            )

            self.export_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败: {str(e)}")

    def _format_prediction_result(self, predictions, model):
        """格式化预测结果"""
        result_df = self.predict_data.copy()
        result_df[self.target] = predictions

        # 添加概率（分类模型）
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(self.predict_data)
            proba_df = pd.DataFrame(
                proba,
                columns=[f"类别_{i}_概率" for i in range(proba.shape[1])]
            )
            result_df = pd.concat([result_df, proba_df], axis=1)

        return result_df

    def _handle_unsupervised(self, model):
        """处理无监督模型"""
        if hasattr(model, 'transform'):
            transformed = model.transform(self.predict_data)
            return pd.DataFrame(
                transformed,
                columns=[f"成分_{i}" for i in range(transformed.shape[1])]
            )
        else:
            clusters = model.predict(self.predict_data)
            result_df = self.predict_data.copy()
            result_df['聚类结果'] = clusters
            return result_df

    def _calculate_scores(self, predictions, model, problem_type):
        """计算评估得分"""
        score_info = {'score_type': 'N/A', 'score_value': 'N/A'}

        if self.true_labels is not None:
            try:
                if problem_type == 'classification':  # 分类模型
                    accuracy = accuracy_score(self.true_labels, predictions)
                    score_info = {
                        'score_type': '准确率',
                        'score_value': f"{accuracy:.4f}"
                    }
                elif problem_type == 'regression':  # 回归模型
                    r2 = r2_score(self.true_labels, predictions)
                    score_info = {
                        'score_type': 'R²分数',
                        'score_value': f"{r2:.4f}"
                    }
                else:  # 聚类模型
                    if len(predictions) == len(self.true_labels):
                        silhouette = silhouette_score(
                            self.predict_data, predictions)
                        score_info = {
                            'score_type': '轮廓系数',
                            'score_value': f"{silhouette:.4f}"
                        }
            except Exception as e:
                print(f"评分计算失败: {str(e)}")

        return score_info

    def _add_prediction_record(self, timestamp, model_name, sample_count, score_info):
        """添加预测记录"""
        row_position = self.history_table.rowCount()
        self.history_table.insertRow(row_position)

        self.history_table.setItem(row_position, 0, QTableWidgetItem(timestamp))
        self.history_table.setItem(row_position, 1, QTableWidgetItem(model_name))
        self.history_table.setItem(row_position, 2, QTableWidgetItem(str(sample_count)))
        self.history_table.setItem(row_position, 3, QTableWidgetItem(score_info['score_type']))
        self.history_table.setItem(row_position, 4, QTableWidgetItem(score_info['score_value']))

        # 更新得分显示
        if score_info['score_type'] != 'N/A':
            self.score_label.setText(
                f"评估得分：{score_info['score_type']} = {score_info['score_value']}"
            )

    def _display_results(self, result_df):
        """显示预测结果"""
        self.result_table.setRowCount(len(result_df))
        self.result_table.setColumnCount(len(result_df.columns))
        self.result_table.setHorizontalHeaderLabels(result_df.columns)

        for i, row in result_df.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(round(value, 4) if isinstance(value, float) else value))
                self.result_table.setItem(i, j, item)

    def _export_results(self):
        """导出结果"""
        if self.result_table.rowCount() == 0:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存预测结果", "",
            "CSV文件 (*.csv);;Excel文件 (*.xlsx);;所有文件 (*.*)"
        )

        if file_path:
            try:
                # 从表格获取数据
                columns = [self.result_table.horizontalHeaderItem(i).text()
                           for i in range(self.result_table.columnCount())]
                data = []
                for row in range(self.result_table.rowCount()):
                    row_data = [
                        self.result_table.item(row, col).text()
                        for col in range(self.result_table.columnCount())
                    ]
                    data.append(row_data)

                df = pd.DataFrame(data, columns=columns)

                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_excel(file_path, index=False)

                QMessageBox.information(self, "成功", "结果已成功导出")

            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")