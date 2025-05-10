import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal


class DataLoader(QObject):
    """数据加载器，支持多种格式文件加载"""
    data_loaded = pyqtSignal(pd.DataFrame)  # 数据加载完成信号

    def __init__(self):
        super().__init__()

    def load_file(self, file_path):
        """根据文件扩展名自动选择加载方法"""
        if file_path.endswith('.csv'):
            return self.load_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            return self.load_excel(file_path)
        else:
            raise ValueError("Unsupported file format")

    def load_csv(self, file_path):
        """加载CSV文件"""
        try:
            data = pd.read_csv(file_path)
            self.data_loaded.emit(data)
            return data
        except Exception as e:
            raise Exception(f"Failed to load CSV: {str(e)}")

    def load_excel(self, file_path):
        """加载Excel文件"""
        try:
            data = pd.read_excel(file_path)
            self.data_loaded.emit(data)
            return data
        except Exception as e:
            raise Exception(f"Failed to load Excel: {str(e)}")