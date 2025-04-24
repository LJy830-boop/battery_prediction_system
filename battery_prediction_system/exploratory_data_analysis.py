#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池寿命预测模型 - 探索性数据分析
该脚本实现了电池数据的探索性分析，包括数据可视化、统计分析和特征关系分析等。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

class BatteryDataExplorer:
    """
    电池数据探索类，实现数据可视化、统计分析和特征关系分析等功能
    """
    
    def __init__(self, data=None, data_path=None):
        """
        初始化数据探索器
        
        参数:
            data: 数据DataFrame，如果为None则从data_path加载
            data_path: 数据文件路径
        """
        self.data = data
        self.data_path = data_path
        
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
    
    def data_summary(self):
        """
        生成数据摘要，包括基本统计信息、缺失值和数据类型
        
        返回:
            数据摘要字典
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 创建摘要字典
        summary = {}
        
        # 基本信息
        summary['shape'] = self.data.shape
        summary['columns'] = list(self.data.columns)
        summary['dtypes'] = self.data.dtypes.to_dict()
        
        # 描述性统计
        summary['describe'] = self.data.describe()
        
        # 缺失值信息
        missing_values = self.data.isnull().sum()
        missing_percentage = (missing_values / len(self.data)) * 100
        summary['missing_values'] = pd.DataFrame({
            'count': missing_values,
            'percentage': missing_percentage
        })
        
        # 打印摘要信息
        print("数据形状:", summary['shape'])
        print("\n数据列:", summary['columns'])
        print("\n数据类型:\n", pd.Series(summary['dtypes']))
        print("\n描述性统计:\n", summary['describe'])
        print("\n缺失值信息:\n", summary['missing_values'])
        
        # 保存摘要信息到文件
        with open('data_summary.txt', 'w') as f:
            f.write(f"数据形状: {summary['shape']}\n\n")
            f.write(f"数据列: {summary['columns']}\n\n")
            f.write("数据类型:\n")
            for col, dtype in summary['dtypes'].items():
                f.write(f"{col}: {dtype}\n")
            f.write("\n描述性统计:\n")
            f.write(summary['describe'].to_string())
            f.write("\n\n缺失值信息:\n")
            f.write(summary['missing_values'].to_string())
        
        print("\n数据摘要已保存到 'data_summary.txt'")
        
        return summary
    
    def visualize_distributions(self, columns=None):
        """
        可视化数据分布
        
        参数:
            columns: 要可视化的列名列表，如果为None则使用所有数值型列
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 如果未指定列，则使用所有数值型列
        if columns is None:
            columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        # 创建一个新的图形
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 4 * n_cols))
        
        # 如果只有一列，确保axes是二维的
        if n_cols == 1:
            axes = axes.reshape(1, 2)
        
        # 为每一列绘制直方图和箱线图
        for i, col in enumerate(columns):
            # 直方图
            sns.histplot(self.data[col], kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'{col} 分布')
            axes[i, 0].set_xlabel(col)
            axes[i, 0].set_ylabel('频率')
            
            # 箱线图
            sns.boxplot(y=self.data[col], ax=axes[i, 1])
            axes[i, 1].set_title(f'{col} 箱线图')
            axes[i, 1].set_ylabel(col)
        
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('data_distributions.png')
        plt.close()
        
        print("数据分布可视化已保存到 'data_distributions.png'")
    
    def visualize_time_series(self, x_col, y_cols, hue_col=None, title=None):
        """
        可视化时间序列数据
        
        参数:
            x_col: x轴列名（通常是时间或循环次数）
            y_cols: y轴列名列表
            hue_col: 用于分组的列名（可选）
            title: 图表标题（可选）
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 检查列是否存在
        for col in [x_col] + y_cols + ([hue_col] if hue_col else []):
            if col not in self.data.columns:
                raise ValueError(f"列 {col} 不在数据中")
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 6))
        
        # 绘制每个y列的时间序列
        for y_col in y_cols:
            if hue_col:
                # 如果有分组列，为每个分组绘制一条线
                for hue_val, group in self.data.groupby(hue_col):
                    plt.plot(group[x_col], group[y_col], label=f'{y_col} ({hue_val})')
            else:
                # 否则只绘制一条线
                plt.plot(self.data[x_col], self.data[y_col], label=y_col)
        
        plt.xlabel(x_col)
        plt.ylabel('值')
        plt.title(title or f'{", ".join(y_cols)} 随 {x_col} 的变化')
        plt.legend()
        plt.grid(True)
        
        # 保存图形
        filename = f'time_series_{x_col}_{"_".join(y_cols)}.png'
        plt.savefig(filename)
        plt.close()
        
        print(f"时间序列可视化已保存到 '{filename}'")
    
    def visualize_capacity_degradation(self, cycle_col, capacity_col):
        """
        可视化电池容量退化曲线
        
        参数:
            cycle_col: 循环次数列名
            capacity_col: 容量列名
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 检查列是否存在
        for col in [cycle_col, capacity_col]:
            if col not in self.data.columns:
                raise ValueError(f"列 {col} 不在数据中")
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 6))
        
        # 如果数据中有多个电池，按电池ID分组绘制
        if 'battery_id' in self.data.columns:
            for battery_id, group in self.data.groupby('battery_id'):
                # 计算每个循环的最大容量
                capacity_per_cycle = group.groupby(cycle_col)[capacity_col].max()
                plt.plot(capacity_per_cycle.index, capacity_per_cycle.values, label=f'电池 {battery_id}')
        else:
            # 计算每个循环的最大容量
            capacity_per_cycle = self.data.groupby(cycle_col)[capacity_col].max()
            plt.plot(capacity_per_cycle.index, capacity_per_cycle.values)
        
        plt.xlabel('循环次数')
        plt.ylabel('容量 (Ah)')
        plt.title('电池容量退化曲线')
        if 'battery_id' in self.data.columns:
            plt.legend()
        plt.grid(True)
        
        # 添加80%容量线（EOL标准）
        if 'battery_id' not in self.data.columns:
            initial_capacity = capacity_per_cycle.iloc[0]
            eol_capacity = initial_capacity * 0.8
            plt.axhline(y=eol_capacity, color='r', linestyle='--', label='EOL (80% 初始容量)')
            plt.legend()
        
        # 保存图形
        plt.savefig('capacity_degradation.png')
        plt.close()
        
        print("容量退化曲线已保存到 'capacity_degradation.png'")
        
        # 计算SOH并可视化
        plt.figure(figsize=(12, 6))
        
        if 'battery_id' in self.data.columns:
            for battery_id, group in self.data.groupby('battery_id'):
                # 计算每个循环的最大容量
                capacity_per_cycle = group.groupby(cycle_col)[capacity_col].max()
                # 计算SOH = 当前容量 / 初始容量
                initial_capacity = capacity_per_cycle.iloc[0]
                soh = capacity_per_cycle / initial_capacity
                plt.plot(soh.index, soh.values, label=f'电池 {battery_id}')
        else:
            # 计算每个循环的最大容量
            capacity_per_cycle = self.data.groupby(cycle_col)[capacity_col].max()
            # 计算SOH = 当前容量 / 初始容量
            initial_capacity = capacity_per_cycle.iloc[0]
            soh = capacity_per_cycle / initial_capacity
            plt.plot(soh.index, soh.values)
        
        plt.xlabel('循环次数')
        plt.ylabel('SOH')
        plt.title('电池健康状态 (SOH) 曲线')
        if 'battery_id' in self.data.columns:
            plt.legend()
        plt.grid(True)
        
        # 添加EOL线（80% SOH）
        plt.axhline(y=0.8, color='r', linestyle='--', label='EOL (80% SOH)')
        plt.legend()
        
        # 保存图形
        plt.savefig('soh_curve.png')
        plt.close()
        
        print("SOH曲线已保存到 'soh_curve.png'")
    
    def visualize_charge_curves(self, cycle_col, time_col, voltage_col, current_col, cycles_to_plot=None):
        """
        可视化充电曲线
        
        参数:
            cycle_col: 循环次数列名
            time_col: 时间列名
            voltage_col: 电压列名
            current_col: 电流列名
            cycles_to_plot: 要绘制的循环次数列表，如果为None则选择均匀分布的几个循环
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 检查列是否存在
        for col in [cycle_col, time_col, voltage_col, current_col]:
            if col not in self.data.columns:
                raise ValueError(f"列 {col} 不在数据中")
        
        # 获取所有唯一的循环次数
        all_cycles = sorted(self.data[cycle_col].unique())
        
        # 如果未指定要绘制的循环，则选择均匀分布的几个循环
        if cycles_to_plot is None:
            # 选择开始、中间和结束的几个循环
            n_cycles = min(5, len(all_cycles))
            indices = np.linspace(0, len(all_cycles) - 1, n_cycles, dtype=int)
            cycles_to_plot = [all_cycles[i] for i in indices]
        
        # 创建一个新的图形，包含两个子图（电压和电流）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 为每个选定的循环绘制充电曲线
        for cycle in cycles_to_plot:
            # 获取该循环的数据
            cycle_data = self.data[self.data[cycle_col] == cycle]
            
            # 按时间排序
            cycle_data = cycle_data.sort_values(by=time_col)
            
            # 绘制电压曲线
            ax1.plot(cycle_data[time_col], cycle_data[voltage_col], label=f'循环 {cycle}')
            
            # 绘制电流曲线
            ax2.plot(cycle_data[time_col], cycle_data[current_col], label=f'循环 {cycle}')
        
        # 设置标签和标题
        ax1.set_ylabel('电压 (V)')
        ax1.set_title('充电过程中的电压曲线')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_xlabel('时间')
        ax2.set_ylabel('电流 (A)')
        ax2.set_title('充电过程中的电流曲线')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('charge_curves.png')
        plt.close()
        
        print("充电曲线已保存到 'charge_curves.png'")
    
    def visualize_correlation_matrix(self, columns=None):
        """
        可视化相关性矩阵
        
        参数:
            columns: 要包含在相关性矩阵中的列名列表，如果为None则使用所有数值型列
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 如果未指定列，则使用所有数值型列
        if columns is None:
            columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        # 计算相关性矩阵
        corr_matrix = self.data[columns].corr()
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 10))
        
        # 绘制热图
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('特征相关性矩阵')
        
        # 保存图形
        plt.savefig('correlation_matrix.png')
        plt.close()
        
        print("相关性矩阵已保存到 'correlation_matrix.png'")
        
        return corr_matrix
    
    def visualize_pairplot(self, columns=None, hue_col=None):
        """
        可视化特征对之间的关系
        
        参数:
            columns: 要包含在配对图中的列名列表，如果为None则使用所有数值型列
            hue_col: 用于分组的列名（可选）
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 如果未指定列，则使用所有数值型列（限制数量以避免图形过大）
        if columns is None:
            columns = self.data.select_dtypes(include=['float64', 'int64']).columns[:5]
        
        # 创建配对图
        g = sns.pairplot(self.data, vars=columns, hue=hue_col, diag_kind='kde')
        g.fig.suptitle('特征对关系图', y=1.02)
        
        # 保存图形
        plt.savefig('pairplot.png')
        plt.close()
        
        print("特征对关系图已保存到 'pairplot.png'")
    
    def analyze_feature_importance(self, target_col, feature_cols=None):
        """
        分析特征重要性
        
        参数:
            target_col: 目标变量列名
            feature_cols: 特征列名列表，如果为None则使用所有数值型列（除了目标列）
            
        返回:
            特征重要性DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        if target_col not in self.data.columns:
            raise ValueError(f"目标列 {target_col} 不在数据中")
            
        # 如果未指定特征列，则使用所有数值型列（除了目标列）
        if feature_cols is None:
            feature_cols = [col for col in self.data.select_dtypes(include=['float64', 'int64']).columns 
                           if col != target_col]
        
        # 计算每个特征与目标的相关系数
        importance = []
        for col in feature_cols:
            # 计算皮尔逊相关系数
            corr, _ = stats.pearsonr(self.data[col], self.data[target_col])
            importance.append((col, abs(corr)))
        
        # 创建特征重要性DataFrame并按重要性排序
        importance_df = pd.DataFrame(importance, columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # 创建一个新的图形
        plt.figure(figsize=(10, 6))
        
        # 绘制特征重要性条形图
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'特征对 {target_col} 的重要性（基于相关系数）')
        plt.xlabel('重要性（相关系数绝对值）')
        plt.ylabel('特征')
        
        # 保存图形
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("特征重要性分析已保存到 'feature_importance.png'")
        
        return importance_df
    
    def analyze_outliers(self, columns=None, method='zscore', threshold=3):
        """
        分析异常值
        
        参数:
            columns: 要分析的列名列表，如果为None则使用所有数值型列
            method: 异常值检测方法，可选 'zscore'（Z分数）, 'iqr'（四分位距）
            threshold: 异常值阈值，对于Z分数方法，通常为3；对于IQR方法，通常为1.5
            
        返回:
            异常值统计DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 如果未指定列，则使用所有数值型列
        if columns is None:
            columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        
        # 创建异常值统计DataFrame
        outlier_stats = pd.DataFrame(index=columns, columns=['Count', 'Percentage'])
        
        # 对每一列检测异常值
        for col in columns:
            if method == 'zscore':
                # 使用Z分数方法
                z_scores = np.abs(stats.zscore(self.data[col]))
                outliers = z_scores > threshold
            elif method == 'iqr':
                # 使用IQR方法
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (self.data[col] < Q1 - threshold * IQR) | (self.data[col] > Q3 + threshold * IQR)
            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")
            
            # 统计异常值数量和百分比
            outlier_count = outliers.sum()
            outlier_percentage = (outlier_count / len(self.data)) * 100
            
            outlier_stats.loc[col, 'Count'] = outlier_count
            outlier_stats.loc[col, 'Percentage'] = outlier_percentage
        
        # 打印异常值统计
        print(f"使用 {method} 方法（阈值 = {threshold}）检测异常值:")
        print(outlier_stats)
        
        # 保存异常值统计到文件
        outlier_stats.to_csv('outlier_statistics.csv')
        print("异常值统计已保存到 'outlier_statistics.csv'")
        
        return outlier_stats
    
    def analyze_capacity_fade_rate(self, cycle_col, capacity_col):
        """
        分析容量衰减率
        
        参数:
            cycle_col: 循环次数列名
            capacity_col: 容量列名
            
        返回:
            容量衰减率DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 检查列是否存在
        for col in [cycle_col, capacity_col]:
            if col not in self.data.columns:
                raise ValueError(f"列 {col} 不在数据中")
        
        # 计算每个循环的最大容量
        capacity_per_cycle = self.data.groupby(cycle_col)[capacity_col].max().reset_index()
        
        # 计算初始容量
        initial_capacity = capacity_per_cycle[capacity_col].iloc[0]
        
        # 计算SOH
        capacity_per_cycle['SOH'] = capacity_per_cycle[capacity_col] / initial_capacity
        
        # 计算容量衰减率（每个循环的容量损失百分比）
        capacity_per_cycle['fade_rate'] = capacity_per_cycle['SOH'].diff() / capacity_per_cycle[cycle_col].diff() * 100
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 6))
        
        # 绘制容量衰减率
        plt.plot(capacity_per_cycle[cycle_col][1:], capacity_per_cycle['fade_rate'][1:])
        plt.xlabel('循环次数')
        plt.ylabel('容量衰减率 (%/cycle)')
        plt.title('电池容量衰减率')
        plt.grid(True)
        
        # 保存图形
        plt.savefig('capacity_fade_rate.png')
        plt.close()
        
        print("容量衰减率分析已保存到 'capacity_fade_rate.png'")
        
        # 计算平均衰减率
        avg_fade_rate = capacity_per_cycle['fade_rate'][1:].mean()
        print(f"平均容量衰减率: {abs(avg_fade_rate):.6f} %/cycle")
        
        return capacity_per_cycle
    
    def analyze_rul_distribution(self, cycle_col, capacity_col, eol_threshold=0.8):
        """
        分析剩余使用寿命(RUL)分布
        
        参数:
            cycle_col: 循环次数列名
            capacity_col: 容量列名
            eol_threshold: EOL阈值，默认为初始容量的80%
            
        返回:
            RUL DataFrame
        """
        if self.data is None:
            raise ValueError("请先加载数据")
            
        # 检查列是否存在
        for col in [cycle_col, capacity_col]:
            if col not in self.data.columns:
                raise ValueError(f"列 {col} 不在数据中")
        
        # 计算每个循环的最大容量
        capacity_per_cycle = self.data.groupby(cycle_col)[capacity_col].max().reset_index()
        
        # 计算初始容量
        initial_capacity = capacity_per_cycle[capacity_col].iloc[0]
        
        # 计算SOH
        capacity_per_cycle['SOH'] = capacity_per_cycle[capacity_col] / initial_capacity
        
        # 找到EOL循环（SOH首次低于阈值的循环）
        eol_cycle = None
        for i, row in capacity_per_cycle.iterrows():
            if row['SOH'] < eol_threshold:
                eol_cycle = row[cycle_col]
                break
        
        # 如果没有找到EOL循环（所有SOH都高于阈值），则使用最后一个循环
        if eol_cycle is None:
            eol_cycle = capacity_per_cycle[cycle_col].max()
            print(f"警告: 所有SOH都高于阈值 {eol_threshold}，使用最后一个循环 {eol_cycle} 作为EOL")
        else:
            print(f"EOL循环: {eol_cycle} (SOH < {eol_threshold})")
        
        # 计算每个循环的RUL
        capacity_per_cycle['RUL'] = eol_cycle - capacity_per_cycle[cycle_col]
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 6))
        
        # 绘制RUL曲线
        plt.plot(capacity_per_cycle[cycle_col], capacity_per_cycle['RUL'])
        plt.xlabel('循环次数')
        plt.ylabel('剩余使用寿命 (RUL)')
        plt.title('电池剩余使用寿命 (RUL) 曲线')
        plt.grid(True)
        
        # 保存图形
        plt.savefig('rul_curve.png')
        plt.close()
        
        print("RUL分析已保存到 'rul_curve.png'")
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 6))
        
        # 绘制SOH与RUL的关系
        plt.scatter(capacity_per_cycle['SOH'], capacity_per_cycle['RUL'])
        plt.xlabel('健康状态 (SOH)')
        plt.ylabel('剩余使用寿命 (RUL)')
        plt.title('SOH与RUL的关系')
        plt.grid(True)
        
        # 添加趋势线
        z = np.polyfit(capacity_per_cycle['SOH'], capacity_per_cycle['RUL'], 1)
        p = np.poly1d(z)
        plt.plot(capacity_per_cycle['SOH'], p(capacity_per_cycle['SOH']), "r--")
        
        # 保存图形
        plt.savefig('soh_rul_relationship.png')
        plt.close()
        
        print("SOH与RUL关系分析已保存到 'soh_rul_relationship.png'")
        
        return capacity_per_cycle


# 示例用法
if __name__ == "__main__":
    # 创建数据探索器实例
    explorer = BatteryDataExplorer()
    
    # 加载数据（需要替换为实际的数据路径）
    # explorer.load_data("battery_data.csv")
    
    # 生成数据摘要
    # explorer.data_summary()
    
    # 可视化数据分布
    # explorer.visualize_distributions()
    
    # 可视化容量退化曲线
    # explorer.visualize_capacity_degradation('cycle', 'capacity')
    
    # 可视化充电曲线
    # explorer.visualize_charge_curves('cycle', 'time', 'voltage', 'current')
    
    # 可视化相关性矩阵
    # explorer.visualize_correlation_matrix()
    
    # 分析特征重要性
    # explorer.analyze_feature_importance('SOH')
    
    # 分析异常值
    # explorer.analyze_outliers()
    
    # 分析容量衰减率
    # explorer.analyze_capacity_fade_rate('cycle', 'capacity')
    
    # 分析RUL分布
    # explorer.analyze_rul_distribution('cycle', 'capacity')
    
    print("探索性数据分析示例完成")
