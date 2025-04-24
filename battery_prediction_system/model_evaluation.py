#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池寿命预测模型 - 模型评估与优化
该脚本实现了对SOH和RUL预测模型的评估和优化方法。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_score, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from prediction_models import BatteryPredictionModel

class ModelEvaluator:
    """
    模型评估类，实现对SOH和RUL预测模型的评估和优化方法
    """
    
    def __init__(self, features=None, features_path=None):
        """
        初始化模型评估器
        
        参数:
            features: 特征DataFrame，如果为None则从features_path加载
            features_path: 特征文件路径
        """
        self.features = features
        self.features_path = features_path
        self.models = {}
        self.results = {}
        
        # 如果提供了特征路径但没有提供特征，则加载特征
        if self.features is None and self.features_path is not None:
            self.load_features()
    
    def load_features(self, features_path=None):
        """
        加载特征数据
        
        参数:
            features_path: 特征文件路径，如果为None则使用初始化时的路径
            
        返回:
            加载的特征数据
        """
        if features_path is not None:
            self.features_path = features_path
            
        if self.features_path is None:
            raise ValueError("特征路径未指定")
            
        # 根据文件类型选择不同的加载方式
        file_ext = os.path.splitext(self.features_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.features = pd.read_csv(self.features_path)
            elif file_ext in ['.xls', '.xlsx']:
                self.features = pd.read_excel(self.features_path)
            else:
                raise ValueError(f"不支持的文件类型: {file_ext}")
                
            print(f"成功加载特征数据，形状: {self.features.shape}")
            return self.features
            
        except Exception as e:
            print(f"加载特征数据时出错: {str(e)}")
            return None
    
    def prepare_data(self, X_cols, y_col, test_size=0.2, random_state=42, scale=True):
        """
        准备训练和测试数据
        
        参数:
            X_cols: 特征列名列表
            y_col: 目标变量列名
            test_size: 测试集比例
            random_state: 随机种子
            scale: 是否对数据进行标准化
            
        返回:
            (X_train, X_test, y_train, y_test)
        """
        if self.features is None:
            raise ValueError("请先加载特征数据")
            
        # 创建预测模型实例
        predictor = BatteryPredictionModel(features=self.features)
        
        # 准备数据
        X_train, X_test, y_train, y_test = predictor.prepare_data(
            X_cols=X_cols,
            y_col=y_col,
            test_size=test_size,
            random_state=random_state,
            scale=scale
        )
        
        return X_train, X_test, y_train, y_test, predictor
    
    def evaluate_multiple_models(self, X_cols, y_col, test_size=0.2, random_state=42, scale=True):
        """
        评估多个模型
        
        参数:
            X_cols: 特征列名列表
            y_col: 目标变量列名
            test_size: 测试集比例
            random_state: 随机种子
            scale: 是否对数据进行标准化
            
        返回:
            评估结果DataFrame
        """
        if self.features is None:
            raise ValueError("请先加载特征数据")
            
        # 准备数据
        X_train, X_test, y_train, y_test, predictor = self.prepare_data(
            X_cols=X_cols,
            y_col=y_col,
            test_size=test_size,
            random_state=random_state,
            scale=scale
        )
        
        # 定义要评估的模型
        models = {
            'SVR': {
                'build_func': predictor.build_svr_model,
                'params': {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}
            },
            'Random Forest': {
                'build_func': predictor.build_random_forest_model,
                'params': {'n_estimators': 100, 'max_depth': None}
            },
            'Gradient Boosting': {
                'build_func': predictor.build_gradient_boosting_model,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            },
            'XGBoost': {
                'build_func': predictor.build_xgboost_model,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            },
            'LightGBM': {
                'build_func': predictor.build_lightgbm_model,
                'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}
            },
            'Gaussian Process': {
                'build_func': predictor.build_gaussian_process_model,
                'params': {}
            },
            'Linear Regression': {
                'build_func': predictor.build_linear_model,
                'params': {'model_type': 'linear'}
            },
            'Ridge Regression': {
                'build_func': predictor.build_linear_model,
                'params': {'model_type': 'ridge', 'alpha': 1.0}
            },
            'Lasso Regression': {
                'build_func': predictor.build_linear_model,
                'params': {'model_type': 'lasso', 'alpha': 0.1}
            },
            'Elastic Net': {
                'build_func': predictor.build_linear_model,
                'params': {'model_type': 'elastic', 'alpha': 0.1, 'l1_ratio': 0.5}
            },
            'MLP': {
                'build_func': predictor.build_mlp_model,
                'params': {'input_dim': X_train.shape[1], 'hidden_layers': [64, 32]}
            }
        }
        
        # 评估结果
        results = []
        
        # 评估每个模型
        for model_name, model_info in models.items():
            print(f"\n评估模型: {model_name}")
            
            try:
                # 构建模型
                model_info['build_func'](**model_info['params'])
                
                # 训练模型
                predictor.train_model(X_train, y_train)
                
                # 评估模型
                metrics = predictor.evaluate_model(X_test, y_test, plot=False)
                
                # 保存模型和结果
                self.models[model_name] = predictor.model
                self.results[model_name] = metrics
                
                # 添加到结果列表
                results.append({
                    'Model': model_name,
                    'MSE': metrics['mse'],
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'R²': metrics['r2']
                })
                
                # 保存模型
                os.makedirs('models', exist_ok=True)
                predictor.save_model(f"models/{model_name.replace(' ', '_').lower()}_model.joblib")
                
            except Exception as e:
                print(f"评估模型 {model_name} 时出错: {str(e)}")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 按RMSE排序
        results_df = results_df.sort_values('RMSE')
        
        # 打印结果
        print("\n模型评估结果:")
        print(results_df)
        
        # 保存结果
        results_df.to_csv('model_evaluation_results.csv', index=False)
        print("评估结果已保存到 'model_evaluation_results.csv'")
        
        # 可视化结果
        self.visualize_model_comparison(results_df)
        
        return results_df
    
    def visualize_model_comparison(self, results_df):
        """
        可视化模型比较结果
        
        参数:
            results_df: 评估结果DataFrame
        """
        # 创建一个新的图形
        plt.figure(figsize=(12, 10))
        
        # 绘制RMSE比较
        plt.subplot(2, 2, 1)
        sns.barplot(x='RMSE', y='Model', data=results_df)
        plt.title('模型RMSE比较')
        plt.grid(True, axis='x')
        
        # 绘制MAE比较
        plt.subplot(2, 2, 2)
        sns.barplot(x='MAE', y='Model', data=results_df)
        plt.title('模型MAE比较')
        plt.grid(True, axis='x')
        
        # 绘制R²比较
        plt.subplot(2, 2, 3)
        sns.barplot(x='R²', y='Model', data=results_df)
        plt.title('模型R²比较')
        plt.grid(True, axis='x')
        
        # 绘制MSE比较
        plt.subplot(2, 2, 4)
        sns.barplot(x='MSE', y='Model', data=results_df)
        plt.title('模型MSE比较')
        plt.grid(True, axis='x')
        
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('model_comparison.png')
        plt.close()
        
        print("模型比较可视化已保存到 'model_comparison.png'")
    
    def optimize_best_model(self, X_cols, y_col, model_name, param_grid, test_size=0.2, random_state=42, scale=True, cv=5):
        """
        优化最佳模型
        
        参数:
            X_cols: 特征列名列表
            y_col: 目标变量列名
            model_name: 模型名称
            param_grid: 参数网格
            test_size: 测试集比例
            random_state: 随机种子
            scale: 是否对数据进行标准化
            cv: 交叉验证折数
            
        返回:
            最佳参数和最佳模型
        """
        if self.features is None:
            raise ValueError("请先加载特征数据")
            
        # 准备数据
        X_train, X_test, y_train, y_test, predictor = self.prepare_data(
            X_cols=X_cols,
            y_col=y_col,
            test_size=test_size,
            random_state=random_state,
            scale=scale
        )
        
        # 根据模型名称确定模型类型
        if model_name == 'SVR':
            model_type = 'svr'
        elif model_name == 'Random Forest':
            model_type = 'rf'
        elif model_name == 'Gradient Boosting':
            model_type = 'gb'
        elif model_name == 'XGBoost':
            model_type = 'xgb'
        elif model_name == 'LightGBM':
            model_type = 'lgb'
        elif model_name == 'Linear Regression':
            model_type = 'linear'
        elif model_name == 'Ridge Regression':
            model_type = 'ridge'
        elif model_name == 'Lasso Regression':
            model_type = 'lasso'
        elif model_name == 'Elastic Net':
            model_type = 'elastic'
        else:
            raise ValueError(f"不支持的模型名称: {model_name}")
        
        # 优化超参数
        best_params, best_model = predictor.optimize_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            model_type=model_type,
            param_grid=param_grid,
            cv=cv
        )
        
        # 评估最佳模型
        metrics = predictor.evaluate_model(X_test, y_test)
        
        # 保存最佳模型
        os.makedirs('models', exist_ok=True)
        predictor.save_model(f"models/{model_name.replace(' ', '_').lower()}_optimized_model.joblib")
        
        print(f"最佳模型 ({model_name}) 评估结果:")
        print(f"MSE: {metrics['mse']:.6f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"R²: {metrics['r2']:.6f}")
        
        # 如果是基于树的模型，可视化特征重要性
        if model_type in ['rf', 'gb', 'xgb', 'lgb']:
            predictor.visualize_feature_importance(X_cols)
        
        return best_params, best_model
    
    def cross_validate_model(self, X_cols, y_col, model_name, n_splits=5, scale=True):
        """
        交叉验证模型
        
        参数:
            X_cols: 特征列名列表
            y_col: 目标变量列名
            model_name: 模型名称
            n_splits: 交叉验证折数
            scale: 是否对数据进行标准化
            
        返回:
            交叉验证分数
        """
        if self.features is None:
            raise ValueError("请先加载特征数据")
            
        # 分离特征和目标
        X = self.features[X_cols]
        y = self.features[y_col]
        
        # 数据标准化
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # 加载模型
        model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")
        
        model = joblib.load(model_path)
        
        # 创建K折交叉验证
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # 计算交叉验证分数
        mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(mse_scores)
        mae_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        
        # 打印交叉验证结果
        print(f"\n{model_name} 交叉验证结果 ({n_splits} 折):")
        print(f"MSE: {np.mean(mse_scores):.6f} ± {np.std(mse_scores):.6f}")
        print(f"RMSE: {np.mean(rmse_scores):.6f} ± {np.std(rmse_scores):.6f}")
        print(f"MAE: {np.mean(mae_scores):.6f} ± {np.std(mae_scores):.6f}")
        print(f"R²: {np.mean(r2_scores):.6f} ± {np.std(r2_scores):.6f}")
        
        # 创建一个新的图形
        plt.figure(figsize=(10, 6))
        
        # 绘制交叉验证分数
        metrics = ['MSE', 'RMSE', 'MAE', 'R²']
        scores = [mse_scores, rmse_scores, mae_scores, r2_scores]
        
        plt.boxplot(scores, labels=metrics)
        plt.title(f'{model_name} 交叉验证分数')
        plt.grid(True)
        
        # 保存图形
        plt.savefig(f'{model_name.replace(" ", "_").lower()}_cross_validation.png')
        plt.close()
        
        print(f"交叉验证可视化已保存到 '{model_name.replace(' ', '_').lower()}_cross_validation.png'")
        
        return {
            'mse': np.mean(mse_scores),
            'rmse': np.mean(rmse_scores),
            'mae': np.mean(mae_scores),
            'r2': np.mean(r2_scores)
        }
    
    def plot_learning_curve(self, X_cols, y_col, model_name, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scale=True):
        """
        绘制学习曲线
        
        参数:
            X_cols: 特征列名列表
            y_col: 目标变量列名
            model_name: 模型名称
            train_sizes: 训练集大小比例
            cv: 交叉验证折数
            scale: 是否对数据进行标准化
        """
        if self.features is None:
            raise ValueError("请先加载特征数据")
            
        # 分离特征和目标
        X = self.features[X_cols]
        y = self.features[y_col]
        
        # 数据标准化
        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # 加载模型
        model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")
        
        model = joblib.load(model_path)
        
        # 计算学习曲线
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, train_sizes=train_sizes, cv=cv, scoring='neg_mean_squared_error'
        )
        
        # 计算均值和标准差
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        # 创建一个新的图形
        plt.figure(figsize=(10, 6))
        
        # 绘制学习曲线
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练集分数")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="验证集分数")
        plt.xlabel("训练样本数")
        plt.ylabel("MSE")
        plt.title(f"{model_name} 学习曲线")
        plt.legend(loc="best")
        
        # 保存图形
        plt.savefig(f'{model_name.replace(" ", "_").lower()}_learning_curve.png')
        plt.close()
        
        print(f"学习曲线已保存到 '{model_name.replace(' ', '_').lower()}_learning_curve.png'")
    
    def evaluate_ensemble_model(self, X_cols, y_col, model_names, weights=None, test_size=0.2, random_state=42, scale=True):
        """
        评估集成模型
        
        参数:
            X_cols: 特征列名列表
            y_col: 目标变量列名
            model_names: 模型名称列表
            weights: 模型权重列表，如果为None则使用相等权重
            test_size: 测试集比例
            random_state: 随机种子
            scale: 是否对数据进行标准化
            
        返回:
            评估指标
        """
        if self.features is None:
            raise ValueError("请先加载特征数据")
            
        # 准备数据
        X_train, X_test, y_train, y_test, predictor = self.prepare_data(
            X_cols=X_cols,
            y_col=y_col,
            test_size=test_size,
            random_state=random_state,
            scale=scale
        )
        
        # 如果未指定权重，则使用相等权重
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        # 确保权重和模型数量相同
        if len(weights) != len(model_names):
            raise ValueError("权重数量必须与模型数量相同")
        
        # 加载模型
        models = []
        for model_name in model_names:
            model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
            if not os.path.exists(model_path):
                raise ValueError(f"模型文件不存在: {model_path}")
            
            model = joblib.load(model_path)
            models.append(model)
        
        # 预测
        y_preds = []
        for model in models:
            y_pred = model.predict(X_test)
            y_preds.append(y_pred)
        
        # 加权平均预测结果
        y_pred_ensemble = np.zeros_like(y_preds[0])
        for i, y_pred in enumerate(y_preds):
            y_pred_ensemble += weights[i] * y_pred
        
        # 如果目标变量已标准化，则需要反标准化
        if predictor.scaler_y is not None:
            y_test_original = predictor.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_ensemble_original = predictor.scaler_y.inverse_transform(y_pred_ensemble.reshape(-1, 1)).flatten()
        else:
            y_test_original = y_test
            y_pred_ensemble_original = y_pred_ensemble
        
        # 计算评估指标
        mse = mean_squared_error(y_test_original, y_pred_ensemble_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_ensemble_original)
        r2 = r2_score(y_test_original, y_pred_ensemble_original)
        
        # 打印评估指标
        print(f"\n集成模型评估结果:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        # 绘制预测结果图
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_original, y_pred_ensemble_original, alpha=0.5)
        plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title('集成模型预测值 vs 实际值')
        plt.grid(True)
        
        # 添加评估指标文本
        plt.text(0.05, 0.95, f'RMSE: {rmse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.6f}',
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
        
        plt.savefig('ensemble_prediction_results.png')
        plt.close()
        
        print("集成模型预测结果图已保存到 'ensemble_prediction_results.png'")
        
        # 返回评估指标
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics
    
    def evaluate_rul_prediction(self, X_cols, y_col, model_name, eol_threshold=0.8, test_size=0.2, random_state=42, scale=True):
        """
        评估RUL预测
        
        参数:
            X_cols: 特征列名列表
            y_col: 目标变量列名（SOH）
            model_name: 模型名称
            eol_threshold: EOL阈值，默认为初始容量的80%
            test_size: 测试集比例
            random_state: 随机种子
            scale: 是否对数据进行标准化
            
        返回:
            评估指标
        """
        if self.features is None:
            raise ValueError("请先加载特征数据")
            
        # 确保数据按循环次数排序
        if 'cycle' in self.features.columns:
            self.features = self.features.sort_values(by='cycle')
        
        # 准备数据
        X_train, X_test, y_train, y_test, predictor = self.prepare_data(
            X_cols=X_cols,
            y_col=y_col,
            test_size=test_size,
            random_state=random_state,
            scale=scale
        )
        
        # 加载模型
        model_path = f"models/{model_name.replace(' ', '_').lower()}_model.joblib"
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")
        
        predictor.load_model(model_path)
        
        # 计算实际RUL
        if 'cycle' in self.features.columns and y_col in self.features.columns:
            # 按循环次数排序
            sorted_features = self.features.sort_values(by='cycle')
            
            # 找到EOL循环（SOH首次低于阈值的循环）
            eol_cycle = None
            for i, row in sorted_features.iterrows():
                if row[y_col] < eol_threshold:
                    eol_cycle = row['cycle']
                    break
            
            # 如果没有找到EOL循环（所有SOH都高于阈值），则使用最后一个循环
            if eol_cycle is None:
                eol_cycle = sorted_features['cycle'].max()
                print(f"警告: 所有SOH都高于阈值 {eol_threshold}，使用最后一个循环 {eol_cycle} 作为EOL")
            else:
                print(f"EOL循环: {eol_cycle} (SOH < {eol_threshold})")
            
            # 计算每个循环的实际RUL
            sorted_features['RUL'] = eol_cycle - sorted_features['cycle']
            
            # 获取测试集对应的循环
            test_indices = self.features.index.difference(self.features.iloc[X_train.shape[0]:].index)
            test_cycles = self.features.loc[test_indices, 'cycle'].values
            
            # 获取测试集对应的实际RUL
            test_rul = []
            for cycle in test_cycles:
                rul = sorted_features[sorted_features['cycle'] == cycle]['RUL'].values[0]
                test_rul.append(rul)
            
            # 预测测试集的SOH
            y_pred = predictor.predict_soh(X_test)
            
            # 预测测试集的RUL
            pred_rul = []
            for i, soh in enumerate(y_pred):
                rul = predictor.predict_rul(X_test[i:i+1], soh, eol_threshold)
                pred_rul.append(rul)
            
            # 计算评估指标
            mse = mean_squared_error(test_rul, pred_rul)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test_rul, pred_rul)
            
            # 打印评估指标
            print(f"\nRUL预测评估结果:")
            print(f"MSE: {mse:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"MAE: {mae:.6f}")
            
            # 绘制RUL预测结果图
            plt.figure(figsize=(10, 6))
            plt.scatter(test_rul, pred_rul, alpha=0.5)
            plt.plot([min(test_rul), max(test_rul)], [min(test_rul), max(test_rul)], 'r--')
            plt.xlabel('实际RUL')
            plt.ylabel('预测RUL')
            plt.title('RUL预测值 vs 实际值')
            plt.grid(True)
            
            # 添加评估指标文本
            plt.text(0.05, 0.95, f'RMSE: {rmse:.6f}\nMAE: {mae:.6f}',
                     transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
            
            plt.savefig('rul_prediction_results.png')
            plt.close()
            
            print("RUL预测结果图已保存到 'rul_prediction_results.png'")
            
            # 绘制RUL随循环次数的变化
            plt.figure(figsize=(12, 6))
            plt.plot(test_cycles, test_rul, 'b-', label='实际RUL')
            plt.plot(test_cycles, pred_rul, 'r--', label='预测RUL')
            plt.xlabel('循环次数')
            plt.ylabel('RUL')
            plt.title('RUL随循环次数的变化')
            plt.legend()
            plt.grid(True)
            
            plt.savefig('rul_vs_cycle.png')
            plt.close()
            
            print("RUL随循环次数的变化图已保存到 'rul_vs_cycle.png'")
            
            # 返回评估指标
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae
            }
            
            return metrics
        else:
            raise ValueError("特征数据中必须包含cycle和SOH列")


# 示例用法
if __name__ == "__main__":
    # 创建模型评估器实例
    evaluator = ModelEvaluator()
    
    # 加载特征数据（需要替换为实际的特征路径）
    # evaluator.load_features("battery_features.csv")
    
    # 评估多个模型
    # results_df = evaluator.evaluate_multiple_models(
    #     X_cols=['feature1', 'feature2', 'feature3'],
    #     y_col='SOH',
    #     test_size=0.2
    # )
    
    # 优化最佳模型
    # param_grid = {
    #     'C': [0.1, 1, 10, 100],
    #     'gamma': ['scale', 'auto', 0.1, 0.01],
    #     'kernel': ['rbf', 'linear', 'poly']
    # }
    # best_params, best_model = evaluator.optimize_best_model(
    #     X_cols=['feature1', 'feature2', 'feature3'],
    #     y_col='SOH',
    #     model_name='SVR',
    #     param_grid=param_grid
    # )
    
    # 交叉验证模型
    # cv_scores = evaluator.cross_validate_model(
    #     X_cols=['feature1', 'feature2', 'feature3'],
    #     y_col='SOH',
    #     model_name='SVR',
    #     n_splits=5
    # )
    
    # 绘制学习曲线
    # evaluator.plot_learning_curve(
    #     X_cols=['feature1', 'feature2', 'feature3'],
    #     y_col='SOH',
    #     model_name='SVR'
    # )
    
    # 评估集成模型
    # ensemble_metrics = evaluator.evaluate_ensemble_model(
    #     X_cols=['feature1', 'feature2', 'feature3'],
    #     y_col='SOH',
    #     model_names=['SVR', 'Random Forest', 'XGBoost'],
    #     weights=[0.4, 0.3, 0.3]
    # )
    
    # 评估RUL预测
    # rul_metrics = evaluator.evaluate_rul_prediction(
    #     X_cols=['feature1', 'feature2', 'feature3'],
    #     y_col='SOH',
    #     model_name='SVR',
    #     eol_threshold=0.8
    # )
    
    print("模型评估与优化示例完成")
