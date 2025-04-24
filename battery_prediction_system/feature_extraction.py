#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池寿命预测模型 - 特征提取方法
该脚本实现了从电池数据中提取健康特征的各种方法，用于SOH和RUL预测。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy import integrate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pywt
import os
import warnings
warnings.filterwarnings('ignore')

class BatteryFeatureExtractor:
    """
    电池特征提取类，实现从电池数据中提取各种健康特征的方法
    """
    
    def __init__(self, data=None, data_path=None):
        """
        初始化特征提取器
        
        参数:
            data: 数据DataFrame，如果为None则从data_path加载
            data_path: 数据文件路径
        """
        self.data = data
        self.data_path = data_path
        self.features = None
        
        # 如果提供了数据路径但没有提供数据，则加载数据
        if self.data is None and self.data_path is not None:
            self.load_data()
    
    def load_data(self, data_path=None):
        """
        加载电池数据
        
        参数:
            data_path: 数据文件路径，如果为None则使用初始化时的路径
            
        返回:
            加载的数据
        """
        if data_path is not None:
            self.data_path = data_path
            
        if self.data_path is None:
            raise ValueError("数据路径未指定")
            
        # 根据文件类型选择不同的加载方式
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.data = pd.read_csv(self.data_path)
            elif file_ext in ['.xls', '.xlsx']:
                self.data = pd.read_excel(self.data_path)
            elif file_ext == '.mat':
                # 对于MATLAB文件，需要使用scipy.io
                from scipy.io import loadmat
                mat_data = loadmat(self.data_path)
                # 转换为DataFrame，需要根据实际数据结构调整
                # 这里假设mat文件中有一个名为'data'的变量
                self.data = pd.DataFrame(mat_data.get('data', mat_data))
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
                
            print(f"成功加载数据，形状: {self.data.shape}")
            return self.data
            
        except Exception as e:
            print(f"加载数据时出错: {str(e)}")
            return None
    
    def extract_time_domain_features(self, cycle_col, voltage_col, current_col, time_col, capacity_col=None, temp_col=None):
        """
        提取时域特征
        
        参数:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            time_col: 时间列名
            capacity_col: 容量列名（可选）
            temp_col: 温度列名（可选）
            
        返回:
            时域特征DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 创建特征DataFrame
        features_df = pd.DataFrame()
        
        # 对每个循环提取特征
        for cycle, cycle_data in self.data.groupby(cycle_col):
            # 确保数据按时间排序
            cycle_data = cycle_data.sort_values(by=time_col)
            
            # 创建特征字典
            features = {'cycle': cycle}
            
            # 1. 充电时间特征
            features['charging_time'] = cycle_data[time_col].max() - cycle_data[time_col].min()
            
            # 2. 电压统计特征
            features['voltage_mean'] = cycle_data[voltage_col].mean()
            features['voltage_std'] = cycle_data[voltage_col].std()
            features['voltage_min'] = cycle_data[voltage_col].min()
            features['voltage_max'] = cycle_data[voltage_col].max()
            features['voltage_range'] = features['voltage_max'] - features['voltage_min']
            
            # 3. 电流统计特征
            features['current_mean'] = cycle_data[current_col].mean()
            features['current_std'] = cycle_data[current_col].std()
            features['current_min'] = cycle_data[current_col].min()
            features['current_max'] = cycle_data[current_col].max()
            features['current_range'] = features['current_max'] - features['current_min']
            
            # 4. 电压变化率特征
            time_values = cycle_data[time_col].values
            voltage_values = cycle_data[voltage_col].values
            
            if len(time_values) > 1:
                # 计算电压对时间的导数
                voltage_diff = np.diff(voltage_values)
                time_diff = np.diff(time_values)
                voltage_rates = voltage_diff / time_diff
                
                features['voltage_rate_mean'] = np.mean(voltage_rates)
                features['voltage_rate_std'] = np.std(voltage_rates)
                features['voltage_rate_max'] = np.max(voltage_rates)
                features['voltage_rate_min'] = np.min(voltage_rates)
            else:
                features['voltage_rate_mean'] = 0
                features['voltage_rate_std'] = 0
                features['voltage_rate_max'] = 0
                features['voltage_rate_min'] = 0
            
            # 5. 电流变化率特征
            current_values = cycle_data[current_col].values
            
            if len(time_values) > 1:
                # 计算电流对时间的导数
                current_diff = np.diff(current_values)
                current_rates = current_diff / time_diff
                
                features['current_rate_mean'] = np.mean(current_rates)
                features['current_rate_std'] = np.std(current_rates)
                features['current_rate_max'] = np.max(current_rates)
                features['current_rate_min'] = np.min(current_rates)
            else:
                features['current_rate_mean'] = 0
                features['current_rate_std'] = 0
                features['current_rate_max'] = 0
                features['current_rate_min'] = 0
            
            # 6. 曲线面积特征
            features['voltage_curve_area'] = integrate.trapezoid(voltage_values, time_values)
            features['current_curve_area'] = integrate.trapezoid(current_values, time_values)
            
            # 7. 如果有容量数据，计算容量特征
            if capacity_col is not None and capacity_col in cycle_data.columns:
                features['capacity_max'] = cycle_data[capacity_col].max()
                features['capacity_mean'] = cycle_data[capacity_col].mean()
            
            # 8. 如果有温度数据，计算温度特征
            if temp_col is not None and temp_col in cycle_data.columns:
                features['temp_mean'] = cycle_data[temp_col].mean()
                features['temp_std'] = cycle_data[temp_col].std()
                features['temp_min'] = cycle_data[temp_col].min()
                features['temp_max'] = cycle_data[temp_col].max()
                features['temp_range'] = features['temp_max'] - features['temp_min']
            
            # 将特征添加到DataFrame
            features_df = pd.concat([features_df, pd.DataFrame([features])], ignore_index=True)
        
        # 计算SOH（如果有容量数据）
        if 'capacity_max' in features_df.columns:
            initial_capacity = features_df['capacity_max'].iloc[0]
            features_df['SOH'] = features_df['capacity_max'] / initial_capacity
        
        self.features = features_df
        print(f"成功提取了 {features_df.shape[1]-1} 个时域特征")
        
        return features_df
    
    def extract_frequency_domain_features(self, cycle_col, voltage_col, current_col, time_col, n_components=5):
        """
        提取频域特征
        
        参数:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            time_col: 时间列名
            n_components: 保留的频域分量数量
            
        返回:
            频域特征DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 创建特征DataFrame
        features_df = pd.DataFrame()
        
        # 对每个循环提取特征
        for cycle, cycle_data in self.data.groupby(cycle_col):
            # 确保数据按时间排序
            cycle_data = cycle_data.sort_values(by=time_col)
            
            # 创建特征字典
            features = {'cycle': cycle}
            
            # 获取电压和电流数据
            voltage_values = cycle_data[voltage_col].values
            current_values = cycle_data[current_col].values
            
            # 1. 电压频谱特征
            if len(voltage_values) > 1:
                # 计算电压的FFT
                voltage_fft = np.abs(np.fft.fft(voltage_values))
                # 只保留前一半（由于对称性）
                voltage_fft = voltage_fft[:len(voltage_fft)//2]
                
                # 保留前n_components个分量作为特征
                for i in range(min(n_components, len(voltage_fft))):
                    features[f'voltage_fft_{i}'] = voltage_fft[i]
                
                # 计算频谱统计特征
                features['voltage_fft_mean'] = np.mean(voltage_fft)
                features['voltage_fft_std'] = np.std(voltage_fft)
                features['voltage_fft_max'] = np.max(voltage_fft)
                features['voltage_fft_energy'] = np.sum(voltage_fft**2)
            
            # 2. 电流频谱特征
            if len(current_values) > 1:
                # 计算电流的FFT
                current_fft = np.abs(np.fft.fft(current_values))
                # 只保留前一半（由于对称性）
                current_fft = current_fft[:len(current_fft)//2]
                
                # 保留前n_components个分量作为特征
                for i in range(min(n_components, len(current_fft))):
                    features[f'current_fft_{i}'] = current_fft[i]
                
                # 计算频谱统计特征
                features['current_fft_mean'] = np.mean(current_fft)
                features['current_fft_std'] = np.std(current_fft)
                features['current_fft_max'] = np.max(current_fft)
                features['current_fft_energy'] = np.sum(current_fft**2)
            
            # 将特征添加到DataFrame
            features_df = pd.concat([features_df, pd.DataFrame([features])], ignore_index=True)
        
        # 如果已经有时域特征，则合并
        if self.features is not None:
            # 确保两个DataFrame有相同的循环
            merged_df = pd.merge(self.features, features_df, on='cycle', how='inner')
            self.features = merged_df
            print(f"成功提取了 {features_df.shape[1]-1} 个频域特征并合并到现有特征")
        else:
            self.features = features_df
            print(f"成功提取了 {features_df.shape[1]-1} 个频域特征")
        
        return features_df
    
    def extract_wavelet_features(self, cycle_col, voltage_col, current_col, time_col, wavelet='db4', level=3):
        """
        提取小波特征
        
        参数:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            time_col: 时间列名
            wavelet: 小波类型
            level: 分解级别
            
        返回:
            小波特征DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 创建特征DataFrame
        features_df = pd.DataFrame()
        
        # 对每个循环提取特征
        for cycle, cycle_data in self.data.groupby(cycle_col):
            # 确保数据按时间排序
            cycle_data = cycle_data.sort_values(by=time_col)
            
            # 创建特征字典
            features = {'cycle': cycle}
            
            # 获取电压和电流数据
            voltage_values = cycle_data[voltage_col].values
            current_values = cycle_data[current_col].values
            
            # 1. 电压小波特征
            if len(voltage_values) > 2**level:
                # 进行小波分解
                coeffs = pywt.wavedec(voltage_values, wavelet, level=level)
                
                # 提取每个级别的统计特征
                for i, coef in enumerate(coeffs):
                    if i == 0:
                        features[f'voltage_wavelet_a{level}_mean'] = np.mean(coef)
                        features[f'voltage_wavelet_a{level}_std'] = np.std(coef)
                        features[f'voltage_wavelet_a{level}_energy'] = np.sum(coef**2)
                    else:
                        features[f'voltage_wavelet_d{level-i+1}_mean'] = np.mean(coef)
                        features[f'voltage_wavelet_d{level-i+1}_std'] = np.std(coef)
                        features[f'voltage_wavelet_d{level-i+1}_energy'] = np.sum(coef**2)
            
            # 2. 电流小波特征
            if len(current_values) > 2**level:
                # 进行小波分解
                coeffs = pywt.wavedec(current_values, wavelet, level=level)
                
                # 提取每个级别的统计特征
                for i, coef in enumerate(coeffs):
                    if i == 0:
                        features[f'current_wavelet_a{level}_mean'] = np.mean(coef)
                        features[f'current_wavelet_a{level}_std'] = np.std(coef)
                        features[f'current_wavelet_a{level}_energy'] = np.sum(coef**2)
                    else:
                        features[f'current_wavelet_d{level-i+1}_mean'] = np.mean(coef)
                        features[f'current_wavelet_d{level-i+1}_std'] = np.std(coef)
                        features[f'current_wavelet_d{level-i+1}_energy'] = np.sum(coef**2)
            
            # 将特征添加到DataFrame
            features_df = pd.concat([features_df, pd.DataFrame([features])], ignore_index=True)
        
        # 如果已经有其他特征，则合并
        if self.features is not None:
            # 确保两个DataFrame有相同的循环
            merged_df = pd.merge(self.features, features_df, on='cycle', how='inner')
            self.features = merged_df
            print(f"成功提取了 {features_df.shape[1]-1} 个小波特征并合并到现有特征")
        else:
            self.features = features_df
            print(f"成功提取了 {features_df.shape[1]-1} 个小波特征")
        
        return features_df
    
    def extract_incremental_features(self, cycle_col, window_size=5):
        """
        提取增量特征（相邻循环之间的变化）
        
        参数:
            cycle_col: 循环次数列名
            window_size: 滑动窗口大小
            
        返回:
            增量特征DataFrame
        """
        if self.features is None:
            raise ValueError("请先提取基本特征")
            
        # 创建一个新的DataFrame来存储增量特征
        incremental_df = self.features.copy()
        
        # 获取所有特征列（除了cycle）
        feature_cols = [col for col in self.features.columns if col != cycle_col]
        
        # 对每个特征计算增量
        for col in feature_cols:
            # 计算相邻循环之间的差值
            incremental_df[f'{col}_diff'] = incremental_df[col].diff()
            
            # 计算滑动窗口内的变化率
            incremental_df[f'{col}_rate'] = incremental_df[col].diff(window_size) / window_size
            
            # 计算滑动窗口内的平均值
            incremental_df[f'{col}_rolling_mean'] = incremental_df[col].rolling(window=window_size, min_periods=1).mean()
            
            # 计算滑动窗口内的标准差
            incremental_df[f'{col}_rolling_std'] = incremental_df[col].rolling(window=window_size, min_periods=1).std()
        
        # 删除NaN值（由于差分和滑动窗口计算导致的）
        incremental_df = incremental_df.fillna(0)
        
        print(f"成功提取了增量特征，特征总数: {incremental_df.shape[1]}")
        
        return incremental_df
    
    def extract_ic_curve_features(self, cycle_col, voltage_col, current_col, capacity_col=None, n_points=10):
        """
        提取增量容量(IC)曲线特征
        
        参数:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            capacity_col: 容量列名（可选）
            n_points: IC曲线采样点数
            
        返回:
            IC曲线特征DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 创建特征DataFrame
        features_df = pd.DataFrame()
        
        # 对每个循环提取特征
        for cycle, cycle_data in self.data.groupby(cycle_col):
            # 确保数据按电压排序
            cycle_data = cycle_data.sort_values(by=voltage_col)
            
            # 创建特征字典
            features = {'cycle': cycle}
            
            # 获取电压和电流数据
            voltage_values = cycle_data[voltage_col].values
            current_values = cycle_data[current_col].values
            
            # 如果有容量数据，计算dQ/dV（增量容量）
            if capacity_col is not None and capacity_col in cycle_data.columns:
                capacity_values = cycle_data[capacity_col].values
                
                # 计算dQ
                dQ = np.diff(capacity_values)
                # 计算dV
                dV = np.diff(voltage_values)
                # 计算dQ/dV（避免除以零）
                dQdV = np.zeros_like(dV)
                mask = dV != 0
                dQdV[mask] = dQ[mask] / dV[mask]
                
                # 对IC曲线进行平滑处理
                dQdV_smooth = signal.savgol_filter(dQdV, 15, 3)
                
                # 在均匀分布的电压点上采样IC曲线
                voltage_range = np.linspace(voltage_values.min(), voltage_values.max(), n_points)
                
                # 对每个采样点，找到最接近的电压值，并提取对应的dQ/dV
                for i, v in enumerate(voltage_range):
                    # 找到最接近的电压索引
                    idx = np.abs(voltage_values[1:] - v).argmin()
                    features[f'IC_{i}'] = dQdV_smooth[idx]
                
                # 提取IC曲线的峰值特征
                peaks, _ = signal.find_peaks(dQdV_smooth)
                if len(peaks) > 0:
                    # 提取最大峰值的位置和高度
                    max_peak_idx = peaks[np.argmax(dQdV_smooth[peaks])]
                    features['IC_max_peak_voltage'] = voltage_values[1:][max_peak_idx]
                    features['IC_max_peak_height'] = dQdV_smooth[max_peak_idx]
                    
                    # 提取峰值数量
                    features['IC_peak_count'] = len(peaks)
                else:
                    features['IC_max_peak_voltage'] = 0
                    features['IC_max_peak_height'] = 0
                    features['IC_peak_count'] = 0
            
            # 将特征添加到DataFrame
            features_df = pd.concat([features_df, pd.DataFrame([features])], ignore_index=True)
        
        # 如果已经有其他特征，则合并
        if self.features is not None:
            # 确保两个DataFrame有相同的循环
            merged_df = pd.merge(self.features, features_df, on='cycle', how='inner')
            self.features = merged_df
            print(f"成功提取了 {features_df.shape[1]-1} 个IC曲线特征并合并到现有特征")
        else:
            self.features = features_df
            print(f"成功提取了 {features_df.shape[1]-1} 个IC曲线特征")
        
        return features_df
    
    def extract_charging_phase_features(self, cycle_col, voltage_col, current_col, time_col, threshold=0.1):
        """
        提取充电阶段特征（恒流充电、恒压充电等）
        
        参数:
            cycle_col: 循环次数列名
            voltage_col: 电压列名
            current_col: 电流列名
            time_col: 时间列名
            threshold: 电流变化阈值，用于检测充电阶段变化
            
        返回:
            充电阶段特征DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 创建特征DataFrame
        features_df = pd.DataFrame()
        
        # 对每个循环提取特征
        for cycle, cycle_data in self.data.groupby(cycle_col):
            # 确保数据按时间排序
            cycle_data = cycle_data.sort_values(by=time_col)
            
            # 创建特征字典
            features = {'cycle': cycle}
            
            # 获取电压、电流和时间数据
            voltage_values = cycle_data[voltage_col].values
            current_values = cycle_data[current_col].values
            time_values = cycle_data[time_col].values
            
            # 检测充电阶段变化
            # 恒流充电(CC)阶段：电流基本恒定
            # 恒压充电(CV)阶段：电压基本恒定，电流逐渐减小
            
            # 计算电流的变化率
            current_diff = np.diff(current_values)
            current_rate = current_diff / np.diff(time_values)
            
            # 使用阈值检测CC和CV阶段
            cc_mask = np.abs(current_rate) < threshold
            cv_mask = ~cc_mask
            
            # 如果检测到CC阶段
            if np.any(cc_mask):
                cc_start_idx = np.where(cc_mask)[0][0]
                cc_end_idx = np.where(cc_mask)[0][-1]
                
                # CC阶段持续时间
                features['cc_duration'] = time_values[cc_end_idx+1] - time_values[cc_start_idx]
                
                # CC阶段平均电流
                features['cc_current_mean'] = np.mean(current_values[cc_start_idx:cc_end_idx+2])
                
                # CC阶段电压变化
                features['cc_voltage_change'] = voltage_values[cc_end_idx+1] - voltage_values[cc_start_idx]
                
                # CC阶段电压变化率
                features['cc_voltage_rate'] = features['cc_voltage_change'] / features['cc_duration']
            else:
                features['cc_duration'] = 0
                features['cc_current_mean'] = 0
                features['cc_voltage_change'] = 0
                features['cc_voltage_rate'] = 0
            
            # 如果检测到CV阶段
            if np.any(cv_mask):
                cv_start_idx = np.where(cv_mask)[0][0]
                cv_end_idx = np.where(cv_mask)[0][-1]
                
                # CV阶段持续时间
                features['cv_duration'] = time_values[cv_end_idx+1] - time_values[cv_start_idx]
                
                # CV阶段平均电压
                features['cv_voltage_mean'] = np.mean(voltage_values[cv_start_idx:cv_end_idx+2])
                
                # CV阶段电流变化
                features['cv_current_change'] = current_values[cv_end_idx+1] - current_values[cv_start_idx]
                
                # CV阶段电流变化率
                features['cv_current_rate'] = features['cv_current_change'] / features['cv_duration']
            else:
                features['cv_duration'] = 0
                features['cv_voltage_mean'] = 0
                features['cv_current_change'] = 0
                features['cv_current_rate'] = 0
            
            # 计算CC和CV阶段的比例
            total_duration = time_values[-1] - time_values[0]
            features['cc_ratio'] = features['cc_duration'] / total_duration if total_duration > 0 else 0
            features['cv_ratio'] = features['cv_duration'] / total_duration if total_duration > 0 else 0
            
            # 将特征添加到DataFrame
            features_df = pd.concat([features_df, pd.DataFrame([features])], ignore_index=True)
        
        # 如果已经有其他特征，则合并
        if self.features is not None:
            # 确保两个DataFrame有相同的循环
            merged_df = pd.merge(self.features, features_df, on='cycle', how='inner')
            self.features = merged_df
            print(f"成功提取了 {features_df.shape[1]-1} 个充电阶段特征并合并到现有特征")
        else:
            self.features = features_df
            print(f"成功提取了 {features_df.shape[1]-1} 个充电阶段特征")
        
        return features_df
    
    def select_features(self, target_col='SOH', method='mic', n_features=10):
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
                corr, _ = stats.pearsonr(X[feature], y)
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
            from sklearn.feature_selection import SelectKBest, f_regression
            
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
        corr, _ = stats.pearsonr(fusion_df['IHF'], fusion_df[target_col])
        
        print(f"特征融合完成，IHF与{target_col}的相关系数: {corr:.4f}")
        
        return fusion_df
    
    def visualize_feature_importance(self, selected_df, target_col='SOH'):
        """
        可视化特征重要性
        
        参数:
            selected_df: 选择的特征DataFrame
            target_col: 目标变量列名
        """
        if selected_df is None:
            raise ValueError("请先选择特征")
            
        # 获取特征列（除了cycle和target_col）
        feature_cols = [col for col in selected_df.columns if col not in ['cycle', target_col]]
        
        # 计算每个特征与目标的相关系数
        correlations = []
        for feature in feature_cols:
            corr, _ = stats.pearsonr(selected_df[feature], selected_df[target_col])
            correlations.append((feature, corr))
        
        # 创建相关系数DataFrame
        corr_df = pd.DataFrame(correlations, columns=['Feature', 'Correlation'])
        corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
        corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
        
        # 创建一个新的图形
        plt.figure(figsize=(10, 6))
        
        # 绘制特征重要性条形图
        bars = plt.barh(corr_df['Feature'], corr_df['Correlation'])
        
        # 为正负相关设置不同颜色
        for i, bar in enumerate(bars):
            if corr_df['Correlation'].iloc[i] < 0:
                bar.set_color('r')
            else:
                bar.set_color('g')
        
        plt.xlabel('相关系数')
        plt.ylabel('特征')
        plt.title(f'特征与{target_col}的相关性')
        plt.grid(True, axis='x')
        
        # 保存图形
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("特征重要性可视化已保存到 'feature_importance.png'")
    
    def visualize_feature_trends(self, selected_df, target_col='SOH', top_n=3):
        """
        可视化重要特征随循环次数的变化趋势
        
        参数:
            selected_df: 选择的特征DataFrame
            target_col: 目标变量列名
            top_n: 显示的顶部特征数量
        """
        if selected_df is None:
            raise ValueError("请先选择特征")
            
        # 获取特征列（除了cycle和target_col）
        feature_cols = [col for col in selected_df.columns if col not in ['cycle', target_col]]
        
        # 计算每个特征与目标的相关系数
        correlations = []
        for feature in feature_cols:
            corr, _ = stats.pearsonr(selected_df[feature], selected_df[target_col])
            correlations.append((feature, abs(corr)))
        
        # 按相关系数绝对值排序
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # 选择相关性最高的top_n个特征
        top_features = [corr[0] for corr in correlations[:top_n]]
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 8))
        
        # 创建主坐标轴
        ax1 = plt.gca()
        
        # 绘制目标变量
        ax1.plot(selected_df['cycle'], selected_df[target_col], 'k-', label=target_col)
        ax1.set_xlabel('循环次数')
        ax1.set_ylabel(target_col, color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        
        # 创建次坐标轴
        ax2 = ax1.twinx()
        
        # 为每个顶部特征绘制趋势线
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for i, feature in enumerate(top_features):
            color = colors[i % len(colors)]
            ax2.plot(selected_df['cycle'], selected_df[feature], f'{color}-', label=feature)
        
        ax2.set_ylabel('特征值', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title(f'顶部{top_n}个特征与{target_col}随循环次数的变化趋势')
        plt.grid(True)
        
        # 保存图形
        plt.savefig('feature_trends.png')
        plt.close()
        
        print("特征趋势可视化已保存到 'feature_trends.png'")


# 示例用法
if __name__ == "__main__":
    # 创建特征提取器实例
    extractor = BatteryFeatureExtractor()
    
    # 加载数据（需要替换为实际的数据路径）
    # extractor.load_data("battery_data.csv")
    
    # 提取时域特征
    # time_features = extractor.extract_time_domain_features(
    #     cycle_col='cycle',
    #     voltage_col='voltage',
    #     current_col='current',
    #     time_col='time',
    #     capacity_col='capacity'
    # )
    
    # 提取频域特征
    # freq_features = extractor.extract_frequency_domain_features(
    #     cycle_col='cycle',
    #     voltage_col='voltage',
    #     current_col='current',
    #     time_col='time'
    # )
    
    # 提取小波特征
    # wavelet_features = extractor.extract_wavelet_features(
    #     cycle_col='cycle',
    #     voltage_col='voltage',
    #     current_col='current',
    #     time_col='time'
    # )
    
    # 提取增量特征
    # incremental_features = extractor.extract_incremental_features(cycle_col='cycle')
    
    # 提取IC曲线特征
    # ic_features = extractor.extract_ic_curve_features(
    #     cycle_col='cycle',
    #     voltage_col='voltage',
    #     current_col='current',
    #     capacity_col='capacity'
    # )
    
    # 提取充电阶段特征
    # phase_features = extractor.extract_charging_phase_features(
    #     cycle_col='cycle',
    #     voltage_col='voltage',
    #     current_col='current',
    #     time_col='time'
    # )
    
    # 特征选择
    # selected_features = extractor.select_features(target_col='SOH', method='mic')
    
    # 特征融合
    # fusion_features = extractor.feature_fusion(selected_features.columns[1:-1], target_col='SOH')
    
    # 可视化特征重要性
    # extractor.visualize_feature_importance(selected_features, target_col='SOH')
    
    # 可视化特征趋势
    # extractor.visualize_feature_trends(selected_features, target_col='SOH')
    
    print("特征提取示例完成")
