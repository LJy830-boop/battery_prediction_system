#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池寿命预测模型 - 数据预处理流程
该脚本实现了电池数据的预处理流程，包括数据加载、清洗、特征提取和选择等步骤。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import integrate
from scipy.stats import pearsonr
import os
import warnings
warnings.filterwarnings('ignore')

class BatteryDataPreprocessor:
    """
    电池数据预处理类，实现数据加载、清洗、特征提取和选择等功能
    """
    
    def __init__(self, data_path=None):
        """
        初始化预处理器
        
        参数:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.labels = None
        self.scaler = None
    
    def load_data(self, data_path=None):
        """
        加载电池数据
        
        参数:
            data_path: 数据文件路径，如果为None则使用初始化时的路径
            
        返回:
            加载的原始数据
        """
        if data_path is not None:
            self.data_path = data_path
            
        if self.data_path is None:
            raise ValueError("数据路径未指定")
            
        # 根据文件类型选择不同的加载方式
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.raw_data = pd.read_csv(self.data_path)
            elif file_ext in ['.xls', '.xlsx']:
                self.raw_data = pd.read_excel(self.data_path)
            elif file_ext == '.mat':
                # 对于MATLAB文件，需要使用scipy.io
                from scipy.io import loadmat
                mat_data = loadmat(self.data_path)
                # 转换为DataFrame，需要根据实际数据结构调整
                # 这里假设mat文件中有一个名为'data'的变量
                self.raw_data = pd.DataFrame(mat_data.get('data', mat_data))
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
                
            print(f"成功加载数据，形状: {self.raw_data.shape}")
            return self.raw_data
            
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            return None
    
    def clean_data(self, drop_na=True, drop_duplicates=True):
        """
        清洗数据，包括处理缺失值和重复值
        
        参数:
            drop_na: 是否删除含有缺失值的行
            drop_duplicates: 是否删除重复行
            
        返回:
            清洗后的数据
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")
            
        # 创建数据副本以避免修改原始数据
        self.processed_data = self.raw_data.copy()
        
        # 记录原始数据形状
        original_shape = self.processed_data.shape
        
        # 处理缺失值
        if drop_na:
            self.processed_data = self.processed_data.dropna()
        else:
            # 对数值型列使用均值填充，对类别型列使用众数填充
            num_cols = self.processed_data.select_dtypes(include=['float64', 'int64']).columns
            cat_cols = self.processed_data.select_dtypes(include=['object', 'category']).columns
            
            for col in num_cols:
                self.processed_data[col].fillna(self.processed_data[col].mean(), inplace=True)
                
            for col in cat_cols:
                self.processed_data[col].fillna(self.processed_data[col].mode()[0], inplace=True)
        
        # 处理重复值
        if drop_duplicates:
            self.processed_data = self.processed_data.drop_duplicates()
        
        # 输出清洗结果
        new_shape = self.processed_data.shape
        print(f"数据清洗完成: 原始形状 {original_shape}, 清洗后形状 {new_shape}")
        print(f"移除了 {original_shape[0] - new_shape[0]} 行数据")
        
        return self.processed_data
    
    def extract_features_from_charge_curve(self, voltage_col, current_col, time_col, capacity_col=None, temp_col=None):
        """
        从充电曲线中提取健康特征
        
        参数:
            voltage_col: 电压列名
            current_col: 电流列名
            time_col: 时间列名
            capacity_col: 容量列名（可选）
            temp_col: 温度列名（可选）
            
        返回:
            提取的特征DataFrame
        """
        if self.processed_data is None:
            raise ValueError("请先清洗数据")
        
        # 创建特征DataFrame
        features_df = pd.DataFrame()
        
        # 1. 提取充电时间特征 (HF1)
        # 充电时间通常与电池健康状态相关，随着电池老化，充电时间可能会增加
        features_df['charging_time'] = self.processed_data.groupby('cycle')[time_col].max() - self.processed_data.groupby('cycle')[time_col].min()
        
        # 2. 提取电压上升率特征 (HF2)
        # 计算每个循环中电压的上升率（斜率）
        def calc_voltage_slope(group):
            # 确保时间是升序的
            sorted_group = group.sort_values(by=time_col)
            # 计算电压对时间的线性回归斜率
            time_values = sorted_group[time_col].values
            voltage_values = sorted_group[voltage_col].values
            
            # 避免除以零
            if len(time_values) <= 1 or np.max(time_values) == np.min(time_values):
                return 0
                
            # 简化的斜率计算
            slope = np.polyfit(time_values, voltage_values, 1)[0]
            return slope
            
        features_df['voltage_slope'] = self.processed_data.groupby('cycle').apply(calc_voltage_slope)
        
        # 3. 提取充电曲线面积特征 (HF3)
        # 计算电压-时间曲线下的面积，使用梯形积分法
        def calc_curve_area(group):
            # 确保时间是升序的
            sorted_group = group.sort_values(by=time_col)
            time_values = sorted_group[time_col].values
            voltage_values = sorted_group[voltage_col].values
            
            # 使用梯形法则计算曲线下面积
            area = integrate.trapezoid(voltage_values, time_values)
            return area
            
        features_df['curve_area'] = self.processed_data.groupby('cycle').apply(calc_curve_area)
        
        # 4. 如果有容量数据，计算SOH
        if capacity_col is not None:
            # 计算每个循环的最大容量
            max_capacity_per_cycle = self.processed_data.groupby('cycle')[capacity_col].max()
            # 获取初始容量（假设第一个循环的容量为初始容量）
            initial_capacity = max_capacity_per_cycle.iloc[0]
            # 计算SOH = 当前容量 / 初始容量
            features_df['SOH'] = max_capacity_per_cycle / initial_capacity
        
        # 5. 如果有温度数据，提取温度相关特征
        if temp_col is not None:
            # 计算每个循环的平均温度
            features_df['avg_temperature'] = self.processed_data.groupby('cycle')[temp_col].mean()
            # 计算每个循环的温度变化范围
            features_df['temp_range'] = self.processed_data.groupby('cycle')[temp_col].max() - self.processed_data.groupby('cycle')[temp_col].min()
        
        # 重置索引，使cycle成为一个列
        features_df = features_df.reset_index()
        
        self.features = features_df
        print(f"成功从充电曲线提取了 {features_df.shape[1]-1} 个特征")
        
        return features_df
    
    def select_features(self, target_col='SOH', method='mic', n_features=3):
        """
        特征选择，选择与目标变量最相关的特征
        
        参数:
            target_col: 目标变量列名
            method: 特征选择方法，可选 'pearson'（皮尔逊相关系数）, 'mic'（最大互信息系数）, 'f_regression'
            n_features: 选择的特征数量
            
        返回:
            选择后的特征DataFrame
        """
        if self.features is None:
            raise ValueError("请先提取特征")
            
        if target_col not in self.features.columns:
            raise ValueError(f"目标列 {target_col} 不在特征集中")
        
        # 分离特征和目标
        X = self.features.drop(columns=[target_col, 'cycle'])
        y = self.features[target_col]
        
        # 特征名称
        feature_names = X.columns
        
        # 根据选择的方法进行特征选择
        if method == 'pearson':
            # 计算每个特征与目标的皮尔逊相关系数
            correlations = []
            for feature in feature_names:
                corr, _ = pearsonr(X[feature], y)
                correlations.append((feature, abs(corr)))
            
            # 按相关系数绝对值排序
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # 选择前n_features个特征
            selected_features = [corr[0] for corr in correlations[:n_features]]
            
        elif method == 'mic':
            # 使用最大互信息系数
            from minepy import MINE
            
            mine = MINE()
            mic_scores = []
            
            for feature in feature_names:
                mine.compute_score(X[feature], y)
                mic_scores.append((feature, mine.mic()))
            
            # 按MIC分数排序
            mic_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 选择前n_features个特征
            selected_features = [score[0] for score in mic_scores[:n_features]]
            
        elif method == 'f_regression':
            # 使用F检验
            selector = SelectKBest(f_regression, k=n_features)
            selector.fit(X, y)
            
            # 获取选择的特征索引
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        # 创建选择后的特征DataFrame
        selected_df = self.features[['cycle'] + selected_features + [target_col]]
        
        print(f"使用 {method} 方法选择了 {len(selected_features)} 个特征: {', '.join(selected_features)}")
        
        return selected_df
    
    def feature_fusion(self, selected_features, target_col='SOH'):
        """
        特征融合，将多个健康特征融合为一个间接健康特征(IHF)
        
        参数:
            selected_features: 选择的特征列表
            target_col: 目标变量列名
            
        返回:
            融合后的特征DataFrame
        """
        if self.features is None:
            raise ValueError("请先提取特征")
            
        # 检查所有选择的特征是否在特征集中
        for feature in selected_features:
            if feature not in self.features.columns:
                raise ValueError(f"特征 {feature} 不在特征集中")
        
        # 创建一个新的DataFrame，包含cycle和选择的特征
        fusion_df = self.features[['cycle'] + selected_features + [target_col]].copy()
        
        # 标准化选择的特征
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(fusion_df[selected_features])
        
        # 将标准化后的特征转换为DataFrame
        scaled_df = pd.DataFrame(scaled_features, columns=selected_features)
        
        # 使用PCA进行特征融合
        pca = PCA(n_components=1)
        ihf = pca.fit_transform(scaled_df)
        
        # 将IHF添加到融合DataFrame
        fusion_df['IHF'] = ihf
        
        # 计算IHF与目标变量的相关系数
        corr, _ = pearsonr(fusion_df['IHF'], fusion_df[target_col])
        
        print(f"特征融合完成，IHF与{target_col}的相关系数: {corr:.4f}")
        
        return fusion_df
    
    def normalize_data(self, method='standard'):
        """
        数据标准化/归一化
        
        参数:
            method: 标准化方法，可选 'standard'（标准化）, 'minmax'（归一化）
            
        返回:
            标准化/归一化后的数据
        """
        if self.processed_data is None:
            raise ValueError("请先清洗数据")
            
        # 选择数值型列
        num_cols = self.processed_data.select_dtypes(include=['float64', 'int64']).columns
        
        # 根据选择的方法进行标准化/归一化
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
            
        # 对数值型列进行标准化/归一化
        self.processed_data[num_cols] = self.scaler.fit_transform(self.processed_data[num_cols])
        
        print(f"使用 {method} 方法完成数据标准化/归一化")
        
        return self.processed_data
    
    def split_data(self, features, target_col='SOH', test_size=0.2, random_state=42):
        """
        划分训练集和测试集
        
        参数:
            features: 特征DataFrame
            target_col: 目标变量列名
            test_size: 测试集比例
            random_state: 随机种子
            
        返回:
            (X_train, X_test, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split
        
        # 分离特征和目标
        X = features.drop(columns=[target_col, 'cycle'])
        y = features[target_col]
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print(f"数据集划分完成: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
        
        return X_train, X_test, y_train, y_test
    
    def visualize_features(self, features=None, target_col='SOH'):
        """
        可视化特征与目标变量的关系
        
        参数:
            features: 特征DataFrame，如果为None则使用self.features
            target_col: 目标变量列名
        """
        if features is None:
            if self.features is None:
                raise ValueError("请先提取特征")
            features = self.features
            
        if target_col not in features.columns:
            raise ValueError(f"目标列 {target_col} 不在特征集中")
        
        # 创建一个新的图形
        plt.figure(figsize=(15, 10))
        
        # 获取除了cycle和target_col之外的所有特征
        feature_cols = [col for col in features.columns if col not in ['cycle', target_col]]
        n_features = len(feature_cols)
        
        # 计算子图的行数和列数
        n_rows = (n_features + 1) // 2
        n_cols = min(2, n_features)
        
        # 绘制每个特征与目标变量的散点图
        for i, feature in enumerate(feature_cols):
            plt.subplot(n_rows, n_cols, i+1)
            plt.scatter(features['cycle'], features[feature], c=features[target_col], cmap='viridis')
            plt.colorbar(label=target_col)
            plt.xlabel('Cycle')
            plt.ylabel(feature)
            plt.title(f'{feature} vs Cycle (colored by {target_col})')
        
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('feature_visualization.png')
        plt.close()
        
        print("特征可视化完成，已保存为 'feature_visualization.png'")
        
        # 计算特征之间的相关性
        plt.figure(figsize=(10, 8))
        corr_matrix = features[feature_cols + [target_col]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('特征相关性热图')
        
        # 保存热图
        plt.savefig('feature_correlation.png')
        plt.close()
        
        print("特征相关性热图已保存为 'feature_correlation.png'")


# 示例用法
if __name__ == "__main__":
    # 创建预处理器实例
    preprocessor = BatteryDataPreprocessor()
    
    # 加载数据（需要替换为实际的数据路径）
    # preprocessor.load_data("battery_data.csv")
    
    # 清洗数据
    # preprocessor.clean_data()
    
    # 提取特征
    # preprocessor.extract_features_from_charge_curve(
    #     voltage_col='voltage',
    #     current_col='current',
    #     time_col='time',
    #     capacity_col='capacity'
    # )
    
    # 特征选择
    # selected_features = preprocessor.select_features(method='mic')
    
    # 特征融合
    # fusion_features = preprocessor.feature_fusion(selected_features.columns[1:-1])
    
    # 数据标准化
    # preprocessor.normalize_data()
    
    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = preprocessor.split_data(fusion_features)
    
    # 可视化特征
    # preprocessor.visualize_features(fusion_features)
    
    print("数据预处理流程示例完成")
