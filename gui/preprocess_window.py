from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QComboBox, QPushButton, QTableWidget,
                             QTableWidgetItem, QHeaderView, QCheckBox, QSpinBox,
                             QWidget, QMessageBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, pyqtSignal
import pandas as pd


class PreprocessWindow(QDialog):
    """数据预处理窗口"""
    preprocessing_done = pyqtSignal(pd.DataFrame)

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowModality(Qt.WindowModal)  # 设置为模态对话框
        self.data = data
        self.setWindowTitle("数据预处理")
        self.setGeometry(200, 200, 800, 600)

        self._setup_ui()
        self._populate_feature_table()
        # 填充目标列选择
        self._populate_target_combobox()

    def _populate_target_combobox(self):
        """填充目标列下拉框"""
        self.target_combo.clear()
        self.target_combo.addItems(self.data.columns.tolist())

        # 默认选择最后一列作为目标列
        self.target_combo.setCurrentIndex(len(self.data.columns) - 1)

    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout()

        # 添加目标列选择组件
        target_group = QGroupBox("目标列设置")
        target_layout = QHBoxLayout()

        target_layout.addWidget(QLabel("选择目标列:"))
        self.target_combo = QComboBox()
        target_layout.addWidget(self.target_combo)

        target_group.setLayout(target_layout)
        layout.addWidget(target_group)

        # 特征信息表格
        self.feature_table = QTableWidget()
        self.feature_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.feature_table.verticalHeader().setVisible(False)
        self.feature_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.feature_table)

        # 添加选择按钮组
        button_group = QGroupBox("选择操作")
        button_layout = QHBoxLayout()

        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self._select_all_features)
        button_layout.addWidget(self.select_all_btn)

        self.unselect_all_btn = QPushButton("全不选")
        self.unselect_all_btn.clicked.connect(self._unselect_all_features)
        button_layout.addWidget(self.unselect_all_btn)

        self.select_numeric_btn = QPushButton("只选数值型")
        self.select_numeric_btn.clicked.connect(self._select_numeric_features)
        button_layout.addWidget(self.select_numeric_btn)

        button_group.setLayout(button_layout)
        layout.addWidget(button_group)

        # 缺失值过滤选项
        self.missing_filter_group = QGroupBox("缺失值过滤")
        missing_filter_layout = QHBoxLayout()

        missing_filter_layout.addWidget(QLabel("过滤缺失值超过(%):"))
        self.missing_threshold_spin = QDoubleSpinBox()
        self.missing_threshold_spin.setRange(0, 100)
        self.missing_threshold_spin.setValue(30)
        self.missing_threshold_spin.setSingleStep(5)
        missing_filter_layout.addWidget(self.missing_threshold_spin)

        self.apply_missing_filter_btn = QPushButton("应用过滤")
        self.apply_missing_filter_btn.clicked.connect(self._apply_missing_filter)
        missing_filter_layout.addWidget(self.apply_missing_filter_btn)

        self.missing_filter_group.setLayout(missing_filter_layout)
        layout.addWidget(self.missing_filter_group)

        # 预处理选项
        options_group = QGroupBox("预处理选项")
        options_layout = QVBoxLayout()

        # 缺失值处理
        missing_group = QGroupBox("缺失值处理")
        missing_layout = QHBoxLayout()

        missing_layout.addWidget(QLabel("数值特征:"))
        self.num_missing_combo = QComboBox()
        self.num_missing_combo.addItems(['mean', 'median', 'constant', 'knn'])
        missing_layout.addWidget(self.num_missing_combo)

        missing_layout.addWidget(QLabel("类别特征:"))
        self.cat_missing_combo = QComboBox()
        self.cat_missing_combo.addItems(['most_frequent', 'constant'])
        missing_layout.addWidget(self.cat_missing_combo)

        missing_group.setLayout(missing_layout)
        options_layout.addWidget(missing_group)

        # 分类特征编码
        encoding_group = QGroupBox("分类特征编码")
        encoding_layout = QHBoxLayout()

        encoding_layout.addWidget(QLabel("编码方法:"))
        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(['None', 'OneHot', 'Label'])
        encoding_layout.addWidget(self.encoding_combo)

        encoding_group.setLayout(encoding_layout)
        options_layout.addWidget(encoding_group)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # 特征缩放
        scaling_group = QGroupBox("特征缩放")
        scaling_layout = QHBoxLayout()

        scaling_layout.addWidget(QLabel("缩放方法:"))
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems(['None', 'standard', 'minmax'])
        scaling_layout.addWidget(self.scaling_combo)

        scaling_group.setLayout(scaling_layout)
        options_layout.addWidget(scaling_group)

        # KNN参数
        self.knn_group = QGroupBox("KNN参数")
        knn_layout = QHBoxLayout()

        knn_layout.addWidget(QLabel("邻居数:"))
        self.knn_spin = QSpinBox()
        self.knn_spin.setRange(1, 20)
        self.knn_spin.setValue(5)
        knn_layout.addWidget(self.knn_spin)

        self.knn_group.setLayout(knn_layout)
        self.knn_group.setVisible(False)
        options_layout.addWidget(self.knn_group)

        # 按钮区域
        button_layout = QHBoxLayout()

        self.apply_btn = QPushButton("应用预处理")
        self.apply_btn.clicked.connect(self._apply_preprocessing)
        button_layout.addWidget(self.apply_btn)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

        # 信号连接
        self.num_missing_combo.currentTextChanged.connect(self._update_knn_visibility)

    def _select_all_features(self):
        """全选所有特征"""
        for i in range(self.feature_table.rowCount()):
            checkbox = self.feature_table.cellWidget(i, 4).findChild(QCheckBox)
            checkbox.setChecked(True)

    def _unselect_all_features(self):
        """全不选所有特征"""
        for i in range(self.feature_table.rowCount()):
            checkbox = self.feature_table.cellWidget(i, 4).findChild(QCheckBox)
            checkbox.setChecked(False)

    def _select_numeric_features(self):
        """只选择数值型特征"""
        for i in range(self.feature_table.rowCount()):
            col_name = self.feature_table.item(i, 0).text()
            dtype = self.feature_table.item(i, 1).text()
            checkbox = self.feature_table.cellWidget(i, 4).findChild(QCheckBox)

            # 检查是否是数值型 (包括整数和浮点数)
            is_numeric = any(t in dtype.lower() for t in ['int', 'float', 'number'])
            checkbox.setChecked(is_numeric)

    def _apply_missing_filter(self):
        """应用缺失值过滤"""
        threshold = self.missing_threshold_spin.value() / 100
        total_rows = len(self.data)

        for i in range(self.feature_table.rowCount()):
            col_name = self.feature_table.item(i, 0).text()
            missing_count = int(self.feature_table.item(i, 2).text())
            missing_ratio = missing_count / total_rows
            checkbox = self.feature_table.cellWidget(i, 4).findChild(QCheckBox)

            # 如果缺失比例超过阈值，取消选择该特征
            if missing_ratio > threshold:
                checkbox.setChecked(False)

    def _update_knn_visibility(self, text):
        """根据选择的缺失值处理方法显示/隐藏KNN参数"""
        self.knn_group.setVisible(text == 'knn')

    def _populate_feature_table(self):
        """填充特征信息表格"""
        self.feature_table.setRowCount(len(self.data.columns))
        self.feature_table.setColumnCount(5)
        self.feature_table.setHorizontalHeaderLabels(
            ["特征名", "类型", "缺失值", "唯一值", "选择"])

        for i, col in enumerate(self.data.columns):
            # 特征名
            self.feature_table.setItem(i, 0, QTableWidgetItem(col))

            # 类型
            dtype = str(self.data[col].dtype)
            self.feature_table.setItem(i, 1, QTableWidgetItem(dtype))

            # 缺失值
            missing = str(self.data[col].isnull().sum())
            self.feature_table.setItem(i, 2, QTableWidgetItem(missing))

            # 唯一值
            unique = str(self.data[col].nunique())
            self.feature_table.setItem(i, 3, QTableWidgetItem(unique))

            # 选择复选框
            checkbox = QCheckBox()
            checkbox.setChecked(False)
            checkbox_widget = QWidget()
            checkbox_layout = QHBoxLayout(checkbox_widget)
            checkbox_layout.addWidget(checkbox)
            checkbox_layout.setAlignment(Qt.AlignCenter)
            checkbox_layout.setContentsMargins(0, 0, 0, 0)
            self.feature_table.setCellWidget(i, 4, checkbox_widget)

    def _apply_preprocessing(self):
        """应用预处理 - 保留原始列名"""
        try:
            # 获取选中的目标列
            target_col = self.target_combo.currentText()

            # 获取选中的特征列（排除目标列）
            selected_cols = []
            for i in range(self.feature_table.rowCount()):
                col_name = self.feature_table.item(i, 0).text()
                checkbox = self.feature_table.cellWidget(i, 4).findChild(QCheckBox)
                if checkbox.isChecked() and col_name != target_col:
                    selected_cols.append(col_name)

            if not selected_cols:
                QMessageBox.warning(self, "警告", "请至少选择一个特征列（不包括目标列）")
                return

            # 获取预处理参数
            num_strategy = self.num_missing_combo.currentText()
            cat_strategy = self.cat_missing_combo.currentText()
            scaling = self.scaling_combo.currentText().lower() if self.scaling_combo.currentText() != 'None' else None
            knn_neighbors = self.knn_spin.value() if num_strategy == 'knn' else 5
            encoding = self.encoding_combo.currentText().lower()

            # 执行预处理
            from core.preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()

            # 分离特征和目标列
            X = self.data[selected_cols]
            y = self.data[target_col]

            preprocessor.detect_feature_types(X)
            # 创建预处理管道
            preprocessor.create_preprocessor(
                num_strategy=num_strategy,
                cat_strategy=cat_strategy,
                scaling=scaling,
                knn_neighbors=knn_neighbors,
                encoding=encoding
            )

            # 执行预处理转换
            processed_X = preprocessor.preprocess_data(X)

            # 处理目标列
            if encoding == 'label' or pd.api.types.is_categorical_dtype(y) or pd.api.types.is_object_dtype(y):
                # 对分类目标进行标签编码
                from sklearn.preprocessing import LabelEncoder
                processed_y = pd.Series(LabelEncoder().fit_transform(y.fillna('Missing')),
                                        name=target_col)
            else:
                # 对数值目标填充缺失值
                from sklearn.impute import SimpleImputer
                imp = SimpleImputer(strategy='mean')
                processed_y = pd.Series(imp.fit_transform(y.to_frame()).ravel(),
                                        name=target_col)

            # 合并特征和目标列
            processed_df = pd.concat([processed_X, processed_y], axis=1)

            # 打印调试信息
            print(f"预处理完成，形状: {processed_df.shape}")
            print(f"特征列名: {processed_df.columns}")
            print(f"目标列名: {target_col}")
            print(f"前5行数据:\n{processed_df.head()}")

            # 发送信号
            self.preprocessing_done.emit(processed_df)
            self.accept()

        except Exception as e:
            import traceback
            error_msg = f"预处理失败: {str(e)}\n\n{traceback.format_exc()}"
            print(error_msg)
            QMessageBox.critical(
                self,
                "错误",
                f"预处理失败:\n{str(e)}\n\n请检查数据格式是否符合要求"
            )