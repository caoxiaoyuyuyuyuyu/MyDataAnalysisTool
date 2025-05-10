from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QTableWidget,
                             QHeaderView, QLabel, QAbstractItemView, QTableWidgetItem)
from PyQt5.QtCore import Qt
import pandas as pd


class AnalysisPanel(QWidget):
    """数据分析结果显示面板"""

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题
        title_label = QLabel("数据分析结果")
        title_label.setStyleSheet("""
            font-size: 14px; 
            font-weight: bold; 
            padding: 5px;
            border-bottom: 1px solid #ccc;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 标签页
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.North)

        # 创建各个分析结果标签页
        self.stats_table = self._create_analysis_table("统计量")
        self.dtypes_table = self._create_analysis_table("数据类型")
        self.unique_table = self._create_analysis_table("唯一值数量")
        self.missing_table = self._create_analysis_table("缺失值数量")

        layout.addWidget(self.tab_widget)

    def _create_analysis_table(self, title):
        """创建分析结果表格"""
        table = QTableWidget()
        table.setEditTriggers(QTableWidget.NoEditTriggers)  # 禁止编辑
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.verticalHeader().setVisible(False)  # 隐藏行号
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.setAlternatingRowColors(True)  # 交替行颜色
        table.setStyleSheet("""
            QTableWidget {
                font-size: 12px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #ddd;
            }
        """)

        self.tab_widget.addTab(table, title)
        return table

    def show_analysis_results(self, results):
        """显示分析结果"""
        # 统计量
        self._display_stats(results['统计量'])

        # 数据类型
        self._display_dtypes(results['数据类型'])

        # 唯一值数量
        self._display_uniques(results['唯一值数量'])

        # 缺失值数量
        self._display_missing(results['缺失值数量'])

    def _display_stats(self, stats_df):
        """显示统计量"""
        self.stats_table.setRowCount(stats_df.shape[0])
        self.stats_table.setColumnCount(stats_df.shape[1] + 1)  # 增加指标名列

        # 设置表头
        headers = ["统计量"] + stats_df.columns.tolist()
        self.stats_table.setHorizontalHeaderLabels(headers)

        # 填充数据
        for i, (stat_name, row) in enumerate(stats_df.iterrows()):
            # 第一列显示统计量名称
            self.stats_table.setItem(i, 0, self._create_table_item(stat_name))

            # 填充各列数据
            for j, (col_name, value) in enumerate(row.items(), start=1):
                self.stats_table.setItem(i, j, self._create_table_item(value))

        # 调整列宽
        self.stats_table.resizeColumnsToContents()

    def _display_dtypes(self, dtypes_series):
        """显示数据类型"""
        self.dtypes_table.setRowCount(1)
        self.dtypes_table.setColumnCount(len(dtypes_series))

        # 设置表头
        self.dtypes_table.setHorizontalHeaderLabels(dtypes_series.index.tolist())

        # 填充数据
        for j, (col_name, dtype) in enumerate(dtypes_series.items()):
            self.dtypes_table.setItem(0, j, self._create_table_item(str(dtype)))

        # 调整列宽
        self.dtypes_table.resizeColumnsToContents()

    def _display_uniques(self, uniques_series):
        """显示唯一值数量"""
        self.unique_table.setRowCount(1)
        self.unique_table.setColumnCount(len(uniques_series))

        # 设置表头
        self.unique_table.setHorizontalHeaderLabels(uniques_series.index.tolist())

        # 填充数据
        for j, (col_name, count) in enumerate(uniques_series.items()):
            self.unique_table.setItem(0, j, self._create_table_item(count))

        # 调整列宽
        self.unique_table.resizeColumnsToContents()

    def _display_missing(self, missing_series):
        """显示缺失值数量"""
        self.missing_table.setRowCount(1)
        self.missing_table.setColumnCount(len(missing_series))

        # 设置表头
        self.missing_table.setHorizontalHeaderLabels(missing_series.index.tolist())

        # 填充数据
        for j, (col_name, count) in enumerate(missing_series.items()):
            self.missing_table.setItem(0, j, self._create_table_item(count))

        # 调整列宽
        self.missing_table.resizeColumnsToContents()

    def _create_table_item(self, value):
        """创建表格项"""
        item = QTableWidgetItem(str(value))

        # 如果是数值，右对齐
        if isinstance(value, (int, float)):
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        else:
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        return item