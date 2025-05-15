import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class DataPreprocessor:
    """数据预处理类"""

    def __init__(self):
        self.preprocessor = None
        self.numeric_features = []
        self.categorical_features = []

    def detect_feature_types(self, data: pd.DataFrame):
        """自动检测数值型和类别型特征"""
        self.numeric_features = data.select_dtypes(include=['number']).columns.tolist()
        self.categorical_features = data.select_dtypes(exclude=['number']).columns.tolist()
        return self.numeric_features, self.categorical_features

    def create_preprocessor(self, num_strategy='mean', cat_strategy='most_frequent',
                            scaling=None, knn_neighbors=5, encoding='none'):
        """创建预处理管道

        参数:
            encoding: 'none' | 'onehot' | 'label'
        """
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=num_strategy))
        ])

        # 根据编码选项创建不同的类别特征处理器
        if encoding.lower() == 'none':
            # 不进行编码，只填充缺失值
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=cat_strategy))
            ])
        elif encoding.lower() == 'label':
            # 标签编码
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=cat_strategy)),
                ('label', LabelEncoder())
            ])
        else:
            # 默认使用OneHot编码
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=cat_strategy)),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

        # 添加特征缩放
        if scaling == 'standard':
            numeric_transformer.steps.append(('scaler', StandardScaler()))
        elif scaling == 'minmax':
            numeric_transformer.steps.append(('scaler', MinMaxScaler()))

        # 使用KNN填充缺失值
        if num_strategy == 'knn':
            numeric_transformer.steps[0] = ('imputer', KNNImputer(n_neighbors=knn_neighbors))

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

        return self.preprocessor

    def preprocess_data(self, data: pd.DataFrame):
        """执行数据预处理，修复形状不匹配问题"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not initialized. Call create_preprocessor first.")

        # 自动检测数值型和类别型特征
        print(self.numeric_features, self.categorical_features)

        # 执行预处理
        processed = self.preprocessor.fit_transform(data)
        print("processed", processed)

        # 如果是稀疏矩阵，转换为密集数组
        if hasattr(processed, 'toarray'):
            processed = processed.toarray()

        # 生成正确的列名
        new_columns = []

        # 1. 处理数值特征列名
        new_columns.extend(self.numeric_features)

        print("data", data)
        print("new_columns",  new_columns)
        print("processed", processed)

        # 2. 处理分类特征列名
        cat_transformer = self.preprocessor.transformers_[1][1]

        if len(cat_transformer.steps) > 1 and hasattr(cat_transformer.steps[-1][1], 'categories_'):
            # OneHot编码情况
            ohe = cat_transformer.steps[-1][1]
            for i, col in enumerate(self.categorical_features):
                if i < len(ohe.categories_):
                    new_columns.extend([f"{col}_{val}" for val in ohe.categories_[i]])
        elif len(cat_transformer.steps) > 1 and isinstance(cat_transformer.steps[-1][1], LabelEncoder):
            # 标签编码情况 - 保留原始列名
            new_columns.extend(self.categorical_features)
        else:
            # 不编码情况 - 保留原始列名
            new_columns.extend(self.categorical_features)

        # 检查列名数量是否匹配
        if len(new_columns) != processed.shape[1]:
            # 如果列名不匹配，使用通用列名
            new_columns = [f"feature_{i}" for i in range(processed.shape[1])]
            import warnings
            warnings.warn(f"列名数量不匹配，使用通用列名。预期{len(new_columns)}列，实际{processed.shape[1]}列")

        # 转换为DataFrame
        processed_df = pd.DataFrame(processed, columns=new_columns)
        print(processed_df.head(5))
        return processed_df