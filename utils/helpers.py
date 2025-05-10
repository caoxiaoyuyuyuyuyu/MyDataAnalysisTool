from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt5.QtCore import Qt
import pandas as pd


def df_to_table(df: pd.DataFrame, table: QTableWidget):
    """将DataFrame显示到QTableWidget中"""
    table.setRowCount(df.shape[0])
    table.setColumnCount(df.shape[1])
    table.setHorizontalHeaderLabels(df.columns)

    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            item = QTableWidgetItem(str(value))

            # 如果是数值，右对齐
            if isinstance(value, (int, float)):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            table.setItem(i, j, item)

            table.resizeColumnsToContents()


def validate_data(df: pd.DataFrame, required_cols: list = None):
    """验证数据是否有效"""
    if df.empty:
        raise ValueError("数据为空")

    if required_cols:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要列: {', '.join(missing_cols)}")

    return True