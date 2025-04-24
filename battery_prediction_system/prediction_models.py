#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池寿命预测模型 - 预测模型构建
该脚本实现了SOH和RUL预测的各种模型，包括传统机器学习模型和深度学习模型。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class BatteryPredictionModel:
    """
    电池预测模型类，实现SOH和RUL预测的各种模型
    """
    
    def __init__(self, features=None, features_path=None):
        """
        初始化预测模型
        
        参数:
            features: 特征DataFrame，如果为None则从features_path加载
            features_path: 特征文件路径
        """
        self.features = features
        self.features_path = features_path
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
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
            
        # 检查列是否存在
        for col in X_cols + [y_col]:
            if col not in self.features.columns:
                raise ValueError(f"列 {col} 不在特征数据中")
        
        # 分离特征和目标
        X = self.features[X_cols]
        y = self.features[y_col]
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        # 数据标准化
        if scale:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            X_train = self.scaler_X.fit_transform(X_train)
            X_test = self.scaler_X.transform(X_test)
            
            y_train = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_test = self.scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
        
        print(f"数据准备完成: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_sequence_data(self, X_cols, y_col, sequence_length=5, test_size=0.2, random_state=42, scale=True):
        """
        准备序列数据（用于LSTM等序列模型）
        
        参数:
            X_cols: 特征列名列表
            y_col: 目标变量列名
            sequence_length: 序列长度
            test_size: 测试集比例
            random_state: 随机种子
            scale: 是否对数据进行标准化
            
        返回:
            (X_train, X_test, y_train, y_test)
        """
        if self.features is None:
            raise ValueError("请先加载特征数据")
            
        # 检查列是否存在
        for col in X_cols + [y_col]:
            if col not in self.features.columns:
                raise ValueError(f"列 {col} 不在特征数据中")
        
        # 确保数据按循环次数排序
        if 'cycle' in self.features.columns:
            self.features = self.features.sort_values(by='cycle')
        
        # 分离特征和目标
        X = self.features[X_cols].values
        y = self.features[y_col].values
        
        # 数据标准化
        if scale:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
            
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # 创建序列数据
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i+sequence_length])
            y_seq.append(y[i+sequence_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=test_size, random_state=random_state)
        
        print(f"序列数据准备完成: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本, 序列长度 {sequence_length}")
        
        return X_train, X_test, y_train, y_test
    
    def build_svr_model(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        """
        构建支持向量回归(SVR)模型
        
        参数:
            kernel: 核函数类型
            C: 正则化参数
            epsilon: epsilon-SVR中的epsilon参数
            gamma: 核系数
            
        返回:
            构建的SVR模型
        """
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        print(f"SVR模型构建完成，参数: kernel={kernel}, C={C}, epsilon={epsilon}, gamma={gamma}")
        return self.model
    
    def build_random_forest_model(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        构建随机森林回归模型
        
        参数:
            n_estimators: 树的数量
            max_depth: 树的最大深度
            min_samples_split: 分裂内部节点所需的最小样本数
            
        返回:
            构建的随机森林模型
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        print(f"随机森林模型构建完成，参数: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
        return self.model
    
    def build_gradient_boosting_model(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        构建梯度提升回归模型
        
        参数:
            n_estimators: 提升阶段的数量
            learning_rate: 学习率
            max_depth: 树的最大深度
            
        返回:
            构建的梯度提升模型
        """
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        print(f"梯度提升模型构建完成，参数: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
        return self.model
    
    def build_xgboost_model(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        构建XGBoost回归模型
        
        参数:
            n_estimators: 提升阶段的数量
            learning_rate: 学习率
            max_depth: 树的最大深度
            
        返回:
            构建的XGBoost模型
        """
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        print(f"XGBoost模型构建完成，参数: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
        return self.model
    
    def build_lightgbm_model(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        构建LightGBM回归模型
        
        参数:
            n_estimators: 提升阶段的数量
            learning_rate: 学习率
            max_depth: 树的最大深度
            
        返回:
            构建的LightGBM模型
        """
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
        print(f"LightGBM模型构建完成，参数: n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}")
        return self.model
    
    def build_gaussian_process_model(self, kernel=None):
        """
        构建高斯过程回归模型
        
        参数:
            kernel: 核函数，如果为None则使用默认核函数
            
        返回:
            构建的高斯过程模型
        """
        if kernel is None:
            # 使用组合核函数：RBF + WhiteKernel
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        
        self.model = GaussianProcessRegressor(kernel=kernel, random_state=42)
        print(f"高斯过程模型构建完成，核函数: {kernel}")
        return self.model
    
    def build_linear_model(self, model_type='linear', alpha=1.0, l1_ratio=0.5):
        """
        构建线性回归模型
        
        参数:
            model_type: 模型类型，可选 'linear'（线性回归）, 'ridge'（岭回归）, 'lasso'（Lasso回归）, 'elastic'（弹性网络）
            alpha: 正则化强度（用于ridge, lasso和elastic）
            l1_ratio: L1正则化比例（用于elastic）
            
        返回:
            构建的线性模型
        """
        if model_type == 'linear':
            self.model = LinearRegression()
            print("线性回归模型构建完成")
        elif model_type == 'ridge':
            self.model = Ridge(alpha=alpha, random_state=42)
            print(f"岭回归模型构建完成，参数: alpha={alpha}")
        elif model_type == 'lasso':
            self.model = Lasso(alpha=alpha, random_state=42)
            print(f"Lasso回归模型构建完成，参数: alpha={alpha}")
        elif model_type == 'elastic':
            self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            print(f"弹性网络模型构建完成，参数: alpha={alpha}, l1_ratio={l1_ratio}")
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        return self.model
    
    def build_mlp_model(self, input_dim, hidden_layers=[64, 32], activation='relu', learning_rate=0.001):
        """
        构建多层感知机(MLP)模型
        
        参数:
            input_dim: 输入维度
            hidden_layers: 隐藏层神经元数量列表
            activation: 激活函数
            learning_rate: 学习率
            
        返回:
            构建的MLP模型
        """
        model = Sequential()
        
        # 添加输入层
        model.add(Dense(hidden_layers[0], input_dim=input_dim, activation=activation))
        
        # 添加隐藏层
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            model.add(Dropout(0.2))  # 添加Dropout以减少过拟合
        
        # 添加输出层
        model.add(Dense(1))
        
        # 编译模型
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        self.model = model
        print(f"MLP模型构建完成，隐藏层: {hidden_layers}, 激活函数: {activation}, 学习率: {learning_rate}")
        
        return self.model
    
    def build_lstm_model(self, input_shape, units=[50, 30], activation='relu', learning_rate=0.001):
        """
        构建长短期记忆(LSTM)模型
        
        参数:
            input_shape: 输入形状 (sequence_length, features)
            units: LSTM层神经元数量列表
            activation: 激活函数
            learning_rate: 学习率
            
        返回:
            构建的LSTM模型
        """
        model = Sequential()
        
        # 添加LSTM层
        model.add(LSTM(units[0], input_shape=input_shape, return_sequences=len(units) > 1))
        model.add(Dropout(0.2))
        
        # 添加额外的LSTM层
        for i, unit in enumerate(units[1:]):
            return_sequences = i < len(units) - 2
            model.add(LSTM(unit, return_sequences=return_sequences))
            model.add(Dropout(0.2))
        
        # 添加全连接层
        model.add(Dense(20, activation=activation))
        
        # 添加输出层
        model.add(Dense(1))
        
        # 编译模型
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        self.model = model
        print(f"LSTM模型构建完成，LSTM单元: {units}, 激活函数: {activation}, 学习率: {learning_rate}")
        
        return self.model
    
    def build_hybrid_model(self, input_dim, sequence_length, lstm_units=50, dense_units=[64, 32], activation='relu', learning_rate=0.001):
        """
        构建混合模型（结合LSTM和MLP）
        
        参数:
            input_dim: 静态特征的输入维度
            sequence_length: 序列长度
            lstm_units: LSTM层神经元数量
            dense_units: 全连接层神经元数量列表
            activation: 激活函数
            learning_rate: 学习率
            
        返回:
            构建的混合模型
        """
        # 序列输入（用于LSTM）
        sequence_input = Input(shape=(sequence_length, input_dim))
        lstm_out = LSTM(lstm_units)(sequence_input)
        
        # 静态特征输入（用于MLP）
        static_input = Input(shape=(input_dim,))
        
        # 合并LSTM输出和静态特征
        merged = Concatenate()([lstm_out, static_input])
        
        # 添加全连接层
        x = merged
        for units in dense_units:
            x = Dense(units, activation=activation)(x)
            x = Dropout(0.2)(x)
        
        # 输出层
        output = Dense(1)(x)
        
        # 创建模型
        model = Model(inputs=[sequence_input, static_input], outputs=output)
        
        # 编译模型
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        
        self.model = model
        print(f"混合模型构建完成，LSTM单元: {lstm_units}, 全连接层: {dense_units}, 激活函数: {activation}, 学习率: {learning_rate}")
        
        return self.model
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=1):
        """
        训练模型
        
        参数:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征（可选）
            y_val: 验证目标（可选）
            epochs: 训练轮数（仅用于深度学习模型）
            batch_size: 批量大小（仅用于深度学习模型）
            verbose: 详细程度
            
        返回:
            训练历史（对于深度学习模型）或训练后的模型
        """
        if self.model is None:
            raise ValueError("请先构建模型")
        
        # 检查模型类型
        if isinstance(self.model, (Sequential, Model)):
            # 深度学习模型
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            
            # 添加早停和模型检查点回调
            callbacks = []
            if validation_data is not None:
                callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))
                callbacks.append(ModelCheckpoint('best_model.h5', save_best_only=True))
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=verbose
            )
            
            print(f"深度学习模型训练完成，训练轮数: {len(history.history['loss'])}")
            return history
        else:
            # 传统机器学习模型
            self.model.fit(X_train, y_train)
            print("传统机器学习模型训练完成")
            return self.model
    
    def evaluate_model(self, X_test, y_test, plot=True):
        """
        评估模型性能
        
        参数:
            X_test: 测试特征
            y_test: 测试目标
            plot: 是否绘制预测结果图
            
        返回:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 如果目标变量已标准化，则需要反标准化
        if self.scaler_y is not None:
            y_test_original = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_original = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_test_original = y_test
            y_pred_original = y_pred
        
        # 计算评估指标
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        # 打印评估指标
        print(f"模型评估结果:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        # 绘制预测结果图
        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test_original, y_pred_original, alpha=0.5)
            plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--')
            plt.xlabel('实际值')
            plt.ylabel('预测值')
            plt.title('预测值 vs 实际值')
            plt.grid(True)
            
            # 添加评估指标文本
            plt.text(0.05, 0.95, f'RMSE: {rmse:.6f}\nMAE: {mae:.6f}\nR²: {r2:.6f}',
                     transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
            
            plt.savefig('prediction_results.png')
            plt.close()
            
            print("预测结果图已保存到 'prediction_results.png'")
        
        # 返回评估指标
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics
    
    def predict_soh(self, X):
        """
        预测SOH
        
        参数:
            X: 输入特征
            
        返回:
            预测的SOH值
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 如果特征已标准化，则需要标准化输入
        if self.scaler_X is not None:
            X = self.scaler_X.transform(X)
        
        # 预测
        y_pred = self.model.predict(X)
        
        # 如果目标变量已标准化，则需要反标准化
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def predict_rul(self, X, current_soh, eol_threshold=0.8):
        """
        预测RUL
        
        参数:
            X: 输入特征
            current_soh: 当前SOH值
            eol_threshold: EOL阈值，默认为初始容量的80%
            
        返回:
            预测的RUL值
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 如果当前SOH已经低于EOL阈值，则RUL为0
        if current_soh < eol_threshold:
            return 0
        
        # 预测SOH
        predicted_soh = self.predict_soh(X)
        
        # 计算SOH的变化率（每个循环的SOH损失）
        if 'cycle' in self.features.columns:
            # 按循环次数排序
            sorted_features = self.features.sort_values(by='cycle')
            
            # 计算SOH的差分
            if 'SOH' in sorted_features.columns:
                soh_diff = sorted_features['SOH'].diff().dropna()
                
                # 计算平均SOH变化率（取绝对值，因为SOH通常是递减的）
                avg_soh_change_rate = abs(soh_diff.mean())
                
                # 计算RUL：(current_SOH - EOL_threshold) / avg_soh_change_rate
                rul = (current_soh - eol_threshold) / avg_soh_change_rate
                
                return max(0, round(rul))
            else:
                raise ValueError("特征数据中没有SOH列")
        else:
            raise ValueError("特征数据中没有cycle列")
    
    def optimize_hyperparameters(self, X_train, y_train, model_type, param_grid, cv=5):
        """
        优化模型超参数
        
        参数:
            X_train: 训练特征
            y_train: 训练目标
            model_type: 模型类型
            param_grid: 参数网格
            cv: 交叉验证折数
            
        返回:
            最佳参数和最佳模型
        """
        # 根据模型类型创建基础模型
        if model_type == 'svr':
            base_model = SVR()
        elif model_type == 'rf':
            base_model = RandomForestRegressor(random_state=42)
        elif model_type == 'gb':
            base_model = GradientBoostingRegressor(random_state=42)
        elif model_type == 'xgb':
            base_model = xgb.XGBRegressor(random_state=42)
        elif model_type == 'lgb':
            base_model = lgb.LGBMRegressor(random_state=42)
        elif model_type == 'linear':
            base_model = LinearRegression()
        elif model_type == 'ridge':
            base_model = Ridge(random_state=42)
        elif model_type == 'lasso':
            base_model = Lasso(random_state=42)
        elif model_type == 'elastic':
            base_model = ElasticNet(random_state=42)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        # 创建网格搜索
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        # 获取最佳参数和最佳模型
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        
        print(f"超参数优化完成，最佳参数: {best_params}")
        print(f"最佳交叉验证分数: {-grid_search.best_score_:.6f} (MSE)")
        
        # 更新模型
        self.model = best_model
        
        return best_params, best_model
    
    def save_model(self, model_path):
        """
        保存模型
        
        参数:
            model_path: 模型保存路径
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 根据模型类型选择不同的保存方式
        if isinstance(self.model, (Sequential, Model)):
            # 保存Keras模型
            self.model.save(model_path)
        else:
            # 保存scikit-learn模型
            joblib.dump(self.model, model_path)
        
        # 如果有标准化器，也保存它们
        if self.scaler_X is not None:
            scaler_X_path = os.path.join(os.path.dirname(model_path), 'scaler_X.joblib')
            joblib.dump(self.scaler_X, scaler_X_path)
        
        if self.scaler_y is not None:
            scaler_y_path = os.path.join(os.path.dirname(model_path), 'scaler_y.joblib')
            joblib.dump(self.scaler_y, scaler_y_path)
        
        print(f"模型已保存到 {model_path}")
    
    def load_model(self, model_path):
        """
        加载模型
        
        参数:
            model_path: 模型加载路径
            
        返回:
            加载的模型
        """
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise ValueError(f"模型文件不存在: {model_path}")
        
        # 根据文件扩展名选择不同的加载方式
        file_ext = os.path.splitext(model_path)[1].lower()
        
        if file_ext in ['.h5', '.keras']:
            # 加载Keras模型
            self.model = tf.keras.models.load_model(model_path)
        else:
            # 加载scikit-learn模型
            self.model = joblib.load(model_path)
        
        # 尝试加载标准化器
        scaler_X_path = os.path.join(os.path.dirname(model_path), 'scaler_X.joblib')
        if os.path.exists(scaler_X_path):
            self.scaler_X = joblib.load(scaler_X_path)
        
        scaler_y_path = os.path.join(os.path.dirname(model_path), 'scaler_y.joblib')
        if os.path.exists(scaler_y_path):
            self.scaler_y = joblib.load(scaler_y_path)
        
        print(f"模型已从 {model_path} 加载")
        
        return self.model
    
    def visualize_training_history(self, history):
        """
        可视化训练历史（仅适用于深度学习模型）
        
        参数:
            history: 训练历史对象
        """
        if not isinstance(history, tf.keras.callbacks.History):
            raise ValueError("history必须是Keras训练历史对象")
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='训练损失')
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='验证损失')
        plt.xlabel('轮数')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        # 绘制MAE曲线
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='训练MAE')
        if 'val_mae' in history.history:
            plt.plot(history.history['val_mae'], label='验证MAE')
        plt.xlabel('轮数')
        plt.ylabel('MAE')
        plt.title('训练和验证MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 保存图形
        plt.savefig('training_history.png')
        plt.close()
        
        print("训练历史可视化已保存到 'training_history.png'")
    
    def visualize_feature_importance(self, X_cols):
        """
        可视化特征重要性（仅适用于基于树的模型）
        
        参数:
            X_cols: 特征列名列表
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 检查模型类型
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("此模型不支持特征重要性可视化")
        
        # 获取特征重要性
        importances = self.model.feature_importances_
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'Feature': X_cols,
            'Importance': importances
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # 创建一个新的图形
        plt.figure(figsize=(10, 6))
        
        # 绘制特征重要性条形图
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.title('特征重要性')
        plt.grid(True, axis='x')
        
        # 保存图形
        plt.savefig('feature_importance.png')
        plt.close()
        
        print("特征重要性可视化已保存到 'feature_importance.png'")
        
        return importance_df
    
    def visualize_prediction_over_cycles(self, X, y_true, cycle_col='cycle'):
        """
        可视化不同循环次数下的预测结果
        
        参数:
            X: 输入特征
            y_true: 真实目标值
            cycle_col: 循环次数列名
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 如果X是DataFrame，提取循环次数和特征
        if isinstance(X, pd.DataFrame):
            if cycle_col not in X.columns:
                raise ValueError(f"列 {cycle_col} 不在特征数据中")
            
            cycles = X[cycle_col].values
            X_features = X.drop(columns=[cycle_col])
        else:
            # 如果X是numpy数组，假设循环次数是按顺序的
            cycles = np.arange(len(X))
            X_features = X
        
        # 预测
        y_pred = self.predict_soh(X_features)
        
        # 创建一个新的图形
        plt.figure(figsize=(12, 6))
        
        # 绘制真实值和预测值
        plt.plot(cycles, y_true, 'b-', label='真实值')
        plt.plot(cycles, y_pred, 'r--', label='预测值')
        plt.xlabel('循环次数')
        plt.ylabel('SOH')
        plt.title('不同循环次数下的SOH预测')
        plt.legend()
        plt.grid(True)
        
        # 保存图形
        plt.savefig('prediction_over_cycles.png')
        plt.close()
        
        print("循环次数预测可视化已保存到 'prediction_over_cycles.png'")
    
    def visualize_rul_prediction(self, X, current_soh, eol_threshold=0.8):
        """
        可视化RUL预测结果
        
        参数:
            X: 输入特征
            current_soh: 当前SOH值
            eol_threshold: EOL阈值，默认为初始容量的80%
        """
        if self.model is None:
            raise ValueError("请先训练模型")
        
        # 预测RUL
        rul = self.predict_rul(X, current_soh, eol_threshold)
        
        # 创建一个新的图形
        plt.figure(figsize=(8, 6))
        
        # 绘制当前SOH和EOL阈值
        plt.bar(['当前SOH', 'EOL阈值'], [current_soh, eol_threshold], color=['g', 'r'])
        plt.axhline(y=eol_threshold, color='r', linestyle='--')
        plt.ylabel('SOH')
        plt.title(f'剩余使用寿命 (RUL) 预测: {rul} 循环')
        
        # 添加RUL文本
        plt.text(0.5, 0.5, f'RUL: {rul} 循环',
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=plt.gca().transAxes,
                 fontsize=16,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 保存图形
        plt.savefig('rul_prediction.png')
        plt.close()
        
        print("RUL预测可视化已保存到 'rul_prediction.png'")
        
        return rul


# 示例用法
if __name__ == "__main__":
    # 创建预测模型实例
    predictor = BatteryPredictionModel()
    
    # 加载特征数据（需要替换为实际的特征路径）
    # predictor.load_features("battery_features.csv")
    
    # 准备数据
    # X_train, X_test, y_train, y_test = predictor.prepare_data(
    #     X_cols=['feature1', 'feature2', 'feature3'],
    #     y_col='SOH',
    #     test_size=0.2
    # )
    
    # 构建模型
    # predictor.build_svr_model()
    # predictor.build_random_forest_model()
    # predictor.build_gradient_boosting_model()
    # predictor.build_xgboost_model()
    # predictor.build_lightgbm_model()
    # predictor.build_gaussian_process_model()
    # predictor.build_linear_model()
    # predictor.build_mlp_model(input_dim=X_train.shape[1])
    
    # 训练模型
    # predictor.train_model(X_train, y_train)
    
    # 评估模型
    # metrics = predictor.evaluate_model(X_test, y_test)
    
    # 预测SOH
    # soh_pred = predictor.predict_soh(X_test)
    
    # 预测RUL
    # rul_pred = predictor.predict_rul(X_test[0:1], current_soh=0.9)
    
    # 保存模型
    # predictor.save_model("models/battery_model.joblib")
    
    # 加载模型
    # predictor.load_model("models/battery_model.joblib")
    
    print("预测模型示例完成")
