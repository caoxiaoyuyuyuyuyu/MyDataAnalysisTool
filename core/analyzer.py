import pandas as pd
from PyQt5.QtCore import QObject


class DataAnalyzer(QObject):
    """数据分析核心类"""

    def __init__(self):
        super().__init__()

    def describe_data(self, data: pd.DataFrame):
        """返回数据集的统计描述"""
        return data.describe().T  # 转置以便每列是一个字段

    def get_column_types(self, data: pd.DataFrame):
        """获取各列数据类型"""
        return data.dtypes

    def get_unique_counts(self, data: pd.DataFrame):
        """获取各列唯一值计数"""
        return data.nunique()

    def get_missing_values(self, data: pd.DataFrame):
        """获取缺失值统计"""
        return data.isnull().sum()

    def get_correlation_matrix(self, data: pd.DataFrame):
        """获取相关系数矩阵"""
        return data.corr()