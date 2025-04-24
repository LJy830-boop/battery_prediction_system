# 电池寿命预测模型搭建指南

## 1. 项目概述

本项目旨在构建一个完整的电池寿命预测模型，用于准确预测锂离子电池的健康状态(SOH)和剩余使用寿命(RUL)。该模型从数据预处理开始，经过探索性数据分析、特征提取、模型构建，最终到模型评估和优化，提供了一个端到端的解决方案。

### 1.1 背景介绍

随着电动汽车、储能系统和便携式电子设备的广泛应用，锂离子电池的性能和寿命预测变得越来越重要。准确预测电池的SOH和RUL可以帮助用户及时维护或更换电池，避免意外故障，同时也可以优化电池管理策略，延长电池使用寿命。

### 1.2 项目目标

- 构建一个准确的电池SOH预测模型
- 构建一个可靠的电池RUL预测模型
- 提供完整的数据预处理、特征提取和模型评估方法
- 优化模型性能，提高预测精度

### 1.3 项目结构

本项目包含以下主要组件：

1. 数据预处理模块：`data_preprocessing_pipeline.py`
2. 探索性数据分析模块：`exploratory_data_analysis.py`
3. 特征提取模块：`feature_extraction.py`
4. 预测模型模块：`prediction_models.py`
5. 模型评估与优化模块：`model_evaluation.py`

## 2. 数据预处理

数据预处理是构建电池寿命预测模型的第一步，包括数据加载、清洗、标准化等操作。

### 2.1 数据加载

首先，我们需要加载电池数据。电池数据通常包含以下信息：

- 循环次数（cycle）
- 时间（time）
- 电压（voltage）
- 电流（current）
- 容量（capacity）
- 温度（temperature，可选）

```python
from data_preprocessing_pipeline import BatteryDataPreprocessor

# 创建预处理器实例
preprocessor = BatteryDataPreprocessor()

# 加载数据
data = preprocessor.load_data("battery_data.csv")
```

### 2.2 数据清洗

数据清洗包括处理缺失值、异常值和重复值等。

```python
# 清洗数据
cleaned_data = preprocessor.clean_data(drop_na=True, drop_duplicates=True)
```

### 2.3 特征提取

从原始数据中提取与电池健康状态相关的特征。

```python
# 提取特征
features = preprocessor.extract_features_from_charge_curve(
    voltage_col='voltage',
    current_col='current',
    time_col='time',
    capacity_col='capacity'
)
```

### 2.4 特征选择

选择与目标变量（SOH或RUL）最相关的特征。

```python
# 特征选择
selected_features = preprocessor.select_features(
    target_col='SOH',
    method='mic',  # 最大互信息系数
    n_features=5
)
```

### 2.5 特征融合

将多个健康特征融合为一个间接健康特征(IHF)。

```python
# 特征融合
fusion_features = preprocessor.feature_fusion(
    selected_features=selected_features.columns[1:-1],
    target_col='SOH'
)
```

### 2.6 数据标准化

对数据进行标准化或归一化处理。

```python
# 数据标准化
normalized_data = preprocessor.normalize_data(method='standard')
```

### 2.7 数据划分

将数据划分为训练集和测试集。

```python
# 划分训练集和测试集
X_train, X_test, y_train, y_test = preprocessor.split_data(
    features=fusion_features,
    target_col='SOH',
    test_size=0.2
)
```

## 3. 探索性数据分析

探索性数据分析(EDA)是理解数据特性和模式的重要步骤，有助于后续的特征提取和模型构建。

### 3.1 数据摘要

生成数据的基本统计信息。

```python
from exploratory_data_analysis import BatteryDataExplorer

# 创建数据探索器实例
explorer = BatteryDataExplorer(data=cleaned_data)

# 生成数据摘要
summary = explorer.data_summary()
```

### 3.2 数据分布可视化

可视化数据分布，包括直方图和箱线图。

```python
# 可视化数据分布
explorer.visualize_distributions(columns=['voltage', 'current', 'capacity'])
```

### 3.3 容量退化曲线

可视化电池容量随循环次数的退化曲线。

```python
# 可视化容量退化曲线
explorer.visualize_capacity_degradation(cycle_col='cycle', capacity_col='capacity')
```

### 3.4 充电曲线

可视化不同循环次数下的充电曲线。

```python
# 可视化充电曲线
explorer.visualize_charge_curves(
    cycle_col='cycle',
    time_col='time',
    voltage_col='voltage',
    current_col='current'
)
```

### 3.5 相关性分析

分析特征之间的相关性。

```python
# 可视化相关性矩阵
corr_matrix = explorer.visualize_correlation_matrix()
```

### 3.6 特征重要性分析

分析特征对目标变量的重要性。

```python
# 分析特征重要性
importance_df = explorer.analyze_feature_importance(target_col='SOH')
```

### 3.7 容量衰减率分析

分析电池容量的衰减率。

```python
# 分析容量衰减率
fade_rate_df = explorer.analyze_capacity_fade_rate(cycle_col='cycle', capacity_col='capacity')
```

### 3.8 RUL分布分析

分析剩余使用寿命(RUL)的分布。

```python
# 分析RUL分布
rul_df = explorer.analyze_rul_distribution(
    cycle_col='cycle',
    capacity_col='capacity',
    eol_threshold=0.8
)
```

## 4. 特征提取

特征提取是从原始数据中提取有用信息的过程，对于提高预测模型的性能至关重要。

### 4.1 时域特征

从时域角度提取特征，如统计特征、变化率特征等。

```python
from feature_extraction import BatteryFeatureExtractor

# 创建特征提取器实例
extractor = BatteryFeatureExtractor(data=cleaned_data)

# 提取时域特征
time_features = extractor.extract_time_domain_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    time_col='time',
    capacity_col='capacity'
)
```

### 4.2 频域特征

从频域角度提取特征，如频谱特征。

```python
# 提取频域特征
freq_features = extractor.extract_frequency_domain_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    time_col='time'
)
```

### 4.3 小波特征

使用小波变换提取特征。

```python
# 提取小波特征
wavelet_features = extractor.extract_wavelet_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    time_col='time'
)
```

### 4.4 增量特征

提取相邻循环之间的变化特征。

```python
# 提取增量特征
incremental_features = extractor.extract_incremental_features(cycle_col='cycle')
```

### 4.5 IC曲线特征

从增量容量(IC)曲线中提取特征。

```python
# 提取IC曲线特征
ic_features = extractor.extract_ic_curve_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    capacity_col='capacity'
)
```

### 4.6 充电阶段特征

从充电阶段（恒流充电、恒压充电等）中提取特征。

```python
# 提取充电阶段特征
phase_features = extractor.extract_charging_phase_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    time_col='time'
)
```

### 4.7 特征选择

选择最相关的特征。

```python
# 特征选择
selected_features = extractor.select_features(
    target_col='SOH',
    method='mic',
    n_features=10
)
```

### 4.8 特征融合

融合多个特征。

```python
# 特征融合
fusion_features = extractor.feature_fusion(
    selected_features=selected_features.columns[1:-1],
    target_col='SOH'
)
```

### 4.9 特征可视化

可视化特征重要性和趋势。

```python
# 可视化特征重要性
extractor.visualize_feature_importance(selected_features, target_col='SOH')

# 可视化特征趋势
extractor.visualize_feature_trends(selected_features, target_col='SOH')
```

## 5. 预测模型构建

构建SOH和RUL预测模型，包括传统机器学习模型和深度学习模型。

### 5.1 数据准备

准备训练和测试数据。

```python
from prediction_models import BatteryPredictionModel

# 创建预测模型实例
predictor = BatteryPredictionModel(features=fusion_features)

# 准备数据
X_train, X_test, y_train, y_test = predictor.prepare_data(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    test_size=0.2
)
```

### 5.2 支持向量回归(SVR)模型

构建SVR模型。

```python
# 构建SVR模型
svr_model = predictor.build_svr_model(
    kernel='rbf',
    C=1.0,
    epsilon=0.1
)

# 训练模型
predictor.train_model(X_train, y_train)

# 评估模型
svr_metrics = predictor.evaluate_model(X_test, y_test)
```

### 5.3 随机森林模型

构建随机森林模型。

```python
# 构建随机森林模型
rf_model = predictor.build_random_forest_model(
    n_estimators=100,
    max_depth=None
)

# 训练模型
predictor.train_model(X_train, y_train)

# 评估模型
rf_metrics = predictor.evaluate_model(X_test, y_test)
```

### 5.4 梯度提升模型

构建梯度提升模型。

```python
# 构建梯度提升模型
gb_model = predictor.build_gradient_boosting_model(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

# 训练模型
predictor.train_model(X_train, y_train)

# 评估模型
gb_metrics = predictor.evaluate_model(X_test, y_test)
```

### 5.5 XGBoost模型

构建XGBoost模型。

```python
# 构建XGBoost模型
xgb_model = predictor.build_xgboost_model(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

# 训练模型
predictor.train_model(X_train, y_train)

# 评估模型
xgb_metrics = predictor.evaluate_model(X_test, y_test)
```

### 5.6 LightGBM模型

构建LightGBM模型。

```python
# 构建LightGBM模型
lgb_model = predictor.build_lightgbm_model(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

# 训练模型
predictor.train_model(X_train, y_train)

# 评估模型
lgb_metrics = predictor.evaluate_model(X_test, y_test)
```

### 5.7 高斯过程回归模型

构建高斯过程回归模型。

```python
# 构建高斯过程回归模型
gpr_model = predictor.build_gaussian_process_model()

# 训练模型
predictor.train_model(X_train, y_train)

# 评估模型
gpr_metrics = predictor.evaluate_model(X_test, y_test)
```

### 5.8 线性回归模型

构建线性回归模型。

```python
# 构建线性回归模型
linear_model = predictor.build_linear_model(model_type='linear')

# 训练模型
predictor.train_model(X_train, y_train)

# 评估模型
linear_metrics = predictor.evaluate_model(X_test, y_test)
```

### 5.9 深度学习模型

构建深度学习模型，如MLP和LSTM。

```python
# 构建MLP模型
mlp_model = predictor.build_mlp_model(
    input_dim=X_train.shape[1],
    hidden_layers=[64, 32]
)

# 训练模型
history = predictor.train_model(X_train, y_train, X_val=X_test, y_val=y_test, epochs=100)

# 评估模型
mlp_metrics = predictor.evaluate_model(X_test, y_test)

# 可视化训练历史
predictor.visualize_training_history(history)
```

```python
# 准备序列数据
X_train_seq, X_test_seq, y_train_seq, y_test_seq = predictor.prepare_sequence_data(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    sequence_length=5
)

# 构建LSTM模型
lstm_model = predictor.build_lstm_model(
    input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
    units=[50, 30]
)

# 训练模型
history = predictor.train_model(X_train_seq, y_train_seq, X_val=X_test_seq, y_val=y_test_seq, epochs=100)

# 评估模型
lstm_metrics = predictor.evaluate_model(X_test_seq, y_test_seq)
```

### 5.10 模型保存和加载

保存和加载模型。

```python
# 保存模型
predictor.save_model("models/xgboost_model.joblib")

# 加载模型
predictor.load_model("models/xgboost_model.joblib")
```

### 5.11 SOH预测

预测SOH。

```python
# 预测SOH
soh_pred = predictor.predict_soh(X_test)
```

### 5.12 RUL预测

预测RUL。

```python
# 预测RUL
rul_pred = predictor.predict_rul(X_test[0:1], current_soh=0.9, eol_threshold=0.8)

# 可视化RUL预测
predictor.visualize_rul_prediction(X_test[0:1], current_soh=0.9, eol_threshold=0.8)
```

## 6. 模型评估与优化

评估和优化模型性能，找出最佳模型。

### 6.1 多模型评估

评估多个模型的性能。

```python
from model_evaluation import ModelEvaluator

# 创建模型评估器实例
evaluator = ModelEvaluator(features=fusion_features)

# 评估多个模型
results_df = evaluator.evaluate_multiple_models(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    test_size=0.2
)
```

### 6.2 超参数优化

优化模型超参数。

```python
# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear', 'poly']
}

# 优化SVR模型
best_params, best_model = evaluator.optimize_best_model(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    model_name='SVR',
    param_grid=param_grid
)
```

### 6.3 交叉验证

进行交叉验证。

```python
# 交叉验证SVR模型
cv_scores = evaluator.cross_validate_model(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    model_name='SVR',
    n_splits=5
)
```

### 6.4 学习曲线分析

分析学习曲线。

```python
# 绘制学习曲线
evaluator.plot_learning_curve(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    model_name='SVR'
)
```

### 6.5 集成模型评估

评估集成模型。

```python
# 评估集成模型
ensemble_metrics = evaluator.evaluate_ensemble_model(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    model_names=['SVR', 'Random Forest', 'XGBoost'],
    weights=[0.4, 0.3, 0.3]
)
```

### 6.6 RUL预测评估

评估RUL预测性能。

```python
# 评估RUL预测
rul_metrics = evaluator.evaluate_rul_prediction(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    model_name='SVR',
    eol_threshold=0.8
)
```

## 7. 完整工作流程

下面是一个完整的电池寿命预测模型构建工作流程。

```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 导入自定义模块
from data_preprocessing_pipeline import BatteryDataPreprocessor
from exploratory_data_analysis import BatteryDataExplorer
from feature_extraction import BatteryFeatureExtractor
from prediction_models import BatteryPredictionModel
from model_evaluation import ModelEvaluator

# 创建输出目录
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

# 1. 数据预处理
print("1. 数据预处理")
preprocessor = BatteryDataPreprocessor()
data = preprocessor.load_data("battery_data.csv")
cleaned_data = preprocessor.clean_data()

# 2. 探索性数据分析
print("\n2. 探索性数据分析")
explorer = BatteryDataExplorer(data=cleaned_data)
summary = explorer.data_summary()
explorer.visualize_distributions()
explorer.visualize_capacity_degradation(cycle_col='cycle', capacity_col='capacity')
explorer.visualize_charge_curves(cycle_col='cycle', time_col='time', voltage_col='voltage', current_col='current')
corr_matrix = explorer.visualize_correlation_matrix()
importance_df = explorer.analyze_feature_importance(target_col='SOH')
fade_rate_df = explorer.analyze_capacity_fade_rate(cycle_col='cycle', capacity_col='capacity')
rul_df = explorer.analyze_rul_distribution(cycle_col='cycle', capacity_col='capacity', eol_threshold=0.8)

# 3. 特征提取
print("\n3. 特征提取")
extractor = BatteryFeatureExtractor(data=cleaned_data)
time_features = extractor.extract_time_domain_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    time_col='time',
    capacity_col='capacity'
)
freq_features = extractor.extract_frequency_domain_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    time_col='time'
)
wavelet_features = extractor.extract_wavelet_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    time_col='time'
)
incremental_features = extractor.extract_incremental_features(cycle_col='cycle')
ic_features = extractor.extract_ic_curve_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    capacity_col='capacity'
)
phase_features = extractor.extract_charging_phase_features(
    cycle_col='cycle',
    voltage_col='voltage',
    current_col='current',
    time_col='time'
)
selected_features = extractor.select_features(target_col='SOH', method='mic', n_features=10)
fusion_features = extractor.feature_fusion(selected_features.columns[1:-1], target_col='SOH')
extractor.visualize_feature_importance(selected_features, target_col='SOH')
extractor.visualize_feature_trends(selected_features, target_col='SOH')

# 保存特征数据
fusion_features.to_csv('output/fusion_features.csv', index=False)

# 4. 预测模型构建
print("\n4. 预测模型构建")
predictor = BatteryPredictionModel(features=fusion_features)
X_train, X_test, y_train, y_test = predictor.prepare_data(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    test_size=0.2
)

# 构建和评估多个模型
models = {
    'SVR': predictor.build_svr_model,
    'Random Forest': predictor.build_random_forest_model,
    'Gradient Boosting': predictor.build_gradient_boosting_model,
    'XGBoost': predictor.build_xgboost_model,
    'LightGBM': predictor.build_lightgbm_model,
    'Gaussian Process': predictor.build_gaussian_process_model,
    'Linear Regression': lambda: predictor.build_linear_model(model_type='linear'),
    'Ridge Regression': lambda: predictor.build_linear_model(model_type='ridge'),
    'Lasso Regression': lambda: predictor.build_linear_model(model_type='lasso'),
    'Elastic Net': lambda: predictor.build_linear_model(model_type='elastic'),
    'MLP': lambda: predictor.build_mlp_model(input_dim=X_train.shape[1])
}

results = []
for name, build_func in models.items():
    print(f"\n构建和评估 {name} 模型")
    build_func()
    predictor.train_model(X_train, y_train)
    metrics = predictor.evaluate_model(X_test, y_test)
    results.append({
        'Model': name,
        'MSE': metrics['mse'],
        'RMSE': metrics['rmse'],
        'MAE': metrics['mae'],
        'R²': metrics['r2']
    })
    predictor.save_model(f"models/{name.replace(' ', '_').lower()}_model.joblib")

# 创建结果DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE')
results_df.to_csv('output/model_results.csv', index=False)
print("\n模型评估结果:")
print(results_df)

# 5. 模型评估与优化
print("\n5. 模型评估与优化")
evaluator = ModelEvaluator(features=fusion_features)

# 获取最佳模型名称
best_model_name = results_df.iloc[0]['Model']
print(f"最佳模型: {best_model_name}")

# 根据最佳模型类型定义参数网格
if best_model_name == 'SVR':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'linear', 'poly']
    }
elif best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'learning_rate': [0.01, 0.1, 0.2] if best_model_name != 'Random Forest' else None
    }
    # 移除None值
    param_grid = {k: v for k, v in param_grid.items() if v is not None}
elif best_model_name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net']:
    param_grid = {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'l1_ratio': [0.1, 0.5, 0.9] if best_model_name == 'Elastic Net' else None
    }
    # 移除None值
    param_grid = {k: v for k, v in param_grid.items() if v is not None}
else:
    param_grid = {}

# 优化最佳模型
if param_grid:
    print(f"\n优化 {best_model_name} 模型")
    best_params, best_model = evaluator.optimize_best_model(
        X_cols=fusion_features.columns[1:-1],
        y_col='SOH',
        model_name=best_model_name,
        param_grid=param_grid
    )

# 交叉验证最佳模型
print(f"\n交叉验证 {best_model_name} 模型")
cv_scores = evaluator.cross_validate_model(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    model_name=best_model_name,
    n_splits=5
)

# 绘制学习曲线
print(f"\n绘制 {best_model_name} 学习曲线")
evaluator.plot_learning_curve(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    model_name=best_model_name
)

# 评估集成模型
print("\n评估集成模型")
top_models = results_df.head(3)['Model'].tolist()
ensemble_metrics = evaluator.evaluate_ensemble_model(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    model_names=top_models,
    weights=[0.5, 0.3, 0.2]
)

# 评估RUL预测
print("\n评估RUL预测")
rul_metrics = evaluator.evaluate_rul_prediction(
    X_cols=fusion_features.columns[1:-1],
    y_col='SOH',
    model_name=best_model_name,
    eol_threshold=0.8
)

print("\n电池寿命预测模型构建完成！")
```

## 8. 结论与建议

### 8.1 结论

通过本项目，我们构建了一个完整的电池寿命预测模型，包括数据预处理、探索性数据分析、特征提取、模型构建和评估优化等步骤。主要结论如下：

1. 特征提取是提高预测精度的关键，从充电曲线中提取的健康特征对SOH和RUL预测具有重要意义。
2. 多种机器学习模型中，XGBoost和随机森林通常表现较好，但具体性能取决于数据特性。
3. 集成多个模型可以进一步提高预测精度。
4. 特征融合技术可以有效减少特征数量，同时保持或提高预测性能。

### 8.2 建议

1. 数据质量：确保电池数据的质量和完整性，包括充分的循环次数和测量频率。
2. 特征工程：尝试更多的特征提取方法，特别是针对电池老化机理的特征。
3. 模型选择：根据具体应用场景和数据特性选择合适的模型，不同模型在不同数据集上的表现可能有所不同。
4. 参数优化：对模型进行充分的超参数优化，以获得最佳性能。
5. 在线学习：考虑实现在线学习机制，使模型能够随着新数据的到来不断更新和改进。

## 9. 参考资料

1. 基于机器学习方法的锂电池剩余寿命预测研究进展. 中国储能网. [https://www.escn.com.cn/news/show-1739847.html](https://www.escn.com.cn/news/show-1739847.html)
2. 基于充电过程的锂电池SOH估计和RUL预测. 储能科学与技术. [https://esst.cip.com.cn/CN/10.19799/j.cnki.2095-4239.2022.0165](https://esst.cip.com.cn/CN/10.19799/j.cnki.2095-4239.2022.0165)
3. 基于多健康特征融合的锂电池SOH和RUL预测. 电源技术. [https://manu70.magtech.com.cn/dyjs/CN/10.3969/j.issn.1002-087X.2023.02.014](https://manu70.magtech.com.cn/dyjs/CN/10.3969/j.issn.1002-087X.2023.02.014)
4. 锂离子电池健康状态估计及寿命预测研究进展综述. [http://www.csee.org.cn/pic/u/cms/www/202502/08112343ajzf.pdf](http://www.csee.org.cn/pic/u/cms/www/202502/08112343ajzf.pdf)
5. 深度学习在Li电池RUL、SOH和电池热管理中的研究进展与应用. [https://blog.csdn.net/qq_25443541/article/details/140524998](https://blog.csdn.net/qq_25443541/article/details/140524998)

## 10. 附录

### 10.1 代码结构

```
battery_prediction/
├── data_preprocessing_pipeline.py  # 数据预处理模块
├── exploratory_data_analysis.py    # 探索性数据分析模块
├── feature_extraction.py           # 特征提取模块
├── prediction_models.py            # 预测模型模块
├── model_evaluation.py             # 模型评估与优化模块
├── main.py                         # 主程序
├── models/                         # 保存的模型
└── output/                         # 输出结果
```

### 10.2 依赖库

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
xgboost
lightgbm
tensorflow
pywt
minepy
joblib
```

### 10.3 安装指南

```bash
# 创建虚拟环境
python -m venv battery_env
source battery_env/bin/activate  # Linux/Mac
battery_env\Scripts\activate     # Windows

# 安装依赖库
pip install numpy pandas matplotlib seaborn scikit-learn scipy xgboost lightgbm tensorflow pywt minepy joblib
```

### 10.4 使用说明

1. 准备数据：确保电池数据包含循环次数、时间、电压、电流和容量等信息。
2. 运行主程序：执行`main.py`脚本，按照工作流程自动完成模型构建和评估。
3. 查看结果：在`output`目录中查看生成的特征数据和模型评估结果，在`models`目录中查看保存的模型。
4. 使用模型：加载保存的模型进行SOH和RUL预测。

```python
# 加载模型进行预测
from prediction_models import BatteryPredictionModel

# 加载模型
predictor = BatteryPredictionModel()
predictor.load_model("models/xgboost_model.joblib")

# 准备特征数据
# ...

# 预测SOH
soh_pred = predictor.predict_soh(X_new)

# 预测RUL
rul_pred = predictor.predict_rul(X_new, current_soh=soh_pred[0], eol_threshold=0.8)
```
