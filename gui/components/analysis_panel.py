from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QTableWidget,
                             QHeaderView, QLabel, QAbstractItemView,
                             QTableWidgetItem, QSpinBox, QHBoxLayout,
                             QPushButton)
from PyQt5.QtCore import Qt
import pandas as pd


class AnalysisPanel(QWidget):
    """数据分析结果显示面板（改进版）"""

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

        # 创建标签页
        self.properties_table = self._create_analysis_table("字段属性")
        self.stats_table = self._create_analysis_table("统计量")
        self.preview_table = self._create_preview_tab()  # 数据预览标签页

        layout.addWidget(self.tab_widget)

    def _create_analysis_table(self, title):
        """创建分析结果表格"""
        table = QTableWidget()
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table.setAlternatingRowColors(True)
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

    def _create_preview_tab(self):
        """创建数据预览标签页"""
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)

        # 预览控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        control_layout.addWidget(QLabel("显示行数:"))
        self.preview_rows_spin = QSpinBox()
        self.preview_rows_spin.setRange(1, 1000)
        self.preview_rows_spin.setValue(10)
        control_layout.addWidget(self.preview_rows_spin)

        self.refresh_btn = QPushButton("刷新预览")
        self.refresh_btn.clicked.connect(self._refresh_preview)
        control_layout.addWidget(self.refresh_btn)

        control_layout.addStretch()
        preview_layout.addWidget(control_panel)

        # 预览表格
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        preview_layout.addWidget(self.preview_table)

        self.tab_widget.addTab(preview_widget, "数据预览")
        return self.preview_table

    def show_analysis_results(self, results):
        """显示分析结果"""
        # 合并字段属性（数据类型、唯一值、缺失值）
        self._display_properties(
            results['数据类型'],
            results['唯一值数量'],
            results['缺失值数量']
        )

        # 显示统计量
        self._display_stats(results['统计量'])

        # 显示数据预览
        if hasattr(self, 'current_data'):
            self._refresh_preview()

    def _display_properties(self, dtypes, uniques, missing):
        """显示合并后的字段属性"""
        columns = dtypes.index.tolist()
        self.properties_table.setRowCount(len(columns))
        self.properties_table.setColumnCount(4)
        self.properties_table.setHorizontalHeaderLabels(
            ["字段名", "数据类型", "唯一值数量", "缺失值数量"])

        for i, col in enumerate(columns):
            # 字段名
            self.properties_table.setItem(i, 0, self._create_table_item(col))
            # 数据类型
            self.properties_table.setItem(i, 1, self._create_table_item(str(dtypes[col])))
            # 唯一值数量
            self.properties_table.setItem(i, 2, self._create_table_item(uniques[col]))
            # 缺失值数量
            self.properties_table.setItem(i, 3, self._create_table_item(missing[col]))

        self.properties_table.resizeColumnsToContents()

    def _display_stats(self, stats_df):
        """显示统计量"""
        self.stats_table.setRowCount(stats_df.shape[0])
        self.stats_table.setColumnCount(stats_df.shape[1] + 1)

        headers = ["统计量"] + stats_df.columns.tolist()
        self.stats_table.setHorizontalHeaderLabels(headers)

        for i, (stat_name, row) in enumerate(stats_df.iterrows()):
            self.stats_table.setItem(i, 0, self._create_table_item(stat_name))
            for j, value in enumerate(row, start=1):
                self.stats_table.setItem(i, j, self._create_table_item(value))

        self.stats_table.resizeColumnsToContents()

    def _refresh_preview(self):
        """刷新数据预览"""
        if not hasattr(self, 'current_data') or self.current_data is None:
            return

        rows = min(self.preview_rows_spin.value(), len(self.current_data))
        self.preview_table.setRowCount(rows)
        self.preview_table.setColumnCount(len(self.current_data.columns))

        # 设置表头
        self.preview_table.setHorizontalHeaderLabels(self.current_data.columns.tolist())

        # 填充数据
        for i in range(rows):
            for j, col in enumerate(self.current_data.columns):
                value = self.current_data.iloc[i, j]
                self.preview_table.setItem(i, j, self._create_table_item(value))

        self.preview_table.resizeColumnsToContents()

    def _create_table_item(self, value):
        """创建表格项"""
        item = QTableWidgetItem(str(value))

        if isinstance(value, (int, float)):
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        else:
            item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        return item

    def set_current_data(self, data):
        """设置当前数据集（用于预览）"""
        self.current_data = data