#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池寿命预测模型 - 主程序
该脚本实现了电池寿命预测模型的完整工作流程，从数据预处理到模型评估。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing_pipeline import BatteryDataPreprocessor
from exploratory_data_analysis import BatteryDataExplorer
from feature_extraction import BatteryFeatureExtractor
from prediction_models import BatteryPredictionModel
from model_evaluation import ModelEvaluator

def main():
    """
    电池寿命预测模型的主函数
    """
    print("电池寿命预测模型构建开始...")
    
    # 创建输出目录
    os.makedirs('output', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 1. 数据预处理
    print("\n1. 数据预处理")
    preprocessor = BatteryDataPreprocessor()
    
    # 提示用户输入数据文件路径
    data_path = input("请输入电池数据文件路径（例如：battery_data.csv）：")
    
    # 加载数据
    data = preprocessor.load_data(data_path)
    
    # 数据清洗
    cleaned_data = preprocessor.clean_data(drop_na=True, drop_duplicates=True)
    print(f"数据清洗完成，清洗后数据形状: {cleaned_data.shape}")
    
    # 保存清洗后的数据
    cleaned_data.to_csv('output/cleaned_data.csv', index=False)
    print("清洗后的数据已保存到 'output/cleaned_data.csv'")
    
    # 2. 探索性数据分析
    print("\n2. 探索性数据分析")
    explorer = BatteryDataExplorer(data=cleaned_data)
    
    # 数据摘要
    summary = explorer.data_summary()
    summary.to_csv('output/data_summary.csv')
    print("数据摘要已保存到 'output/data_summary.csv'")
    
    # 可视化数据分布
    explorer.visualize_distributions()
    print("数据分布可视化已保存")
    
    # 可视化容量退化曲线
    if all(col in cleaned_data.columns for col in ['cycle', 'capacity']):
        explorer.visualize_capacity_degradation(cycle_col='cycle', capacity_col='capacity')
        print("容量退化曲线已保存")
    
    # 可视化充电曲线
    if all(col in cleaned_data.columns for col in ['cycle', 'time', 'voltage', 'current']):
        explorer.visualize_charge_curves(
            cycle_col='cycle',
            time_col='time',
            voltage_col='voltage',
            current_col='current'
        )
        print("充电曲线已保存")
    
    # 相关性分析
    corr_matrix = explorer.visualize_correlation_matrix()
    print("相关性矩阵已保存")
    
    # 特征重要性分析
    if 'SOH' in cleaned_data.columns:
        importance_df = explorer.analyze_feature_importance(target_col='SOH')
        importance_df.to_csv('output/feature_importance.csv', index=False)
        print("特征重要性分析已保存到 'output/feature_importance.csv'")
    
    # 容量衰减率分析
    if all(col in cleaned_data.columns for col in ['cycle', 'capacity']):
        fade_rate_df = explorer.analyze_capacity_fade_rate(cycle_col='cycle', capacity_col='capacity')
        fade_rate_df.to_csv('output/capacity_fade_rate.csv', index=False)
        print("容量衰减率分析已保存到 'output/capacity_fade_rate.csv'")
    
    # RUL分布分析
    if all(col in cleaned_data.columns for col in ['cycle', 'capacity']):
        rul_df = explorer.analyze_rul_distribution(
            cycle_col='cycle',
            capacity_col='capacity',
            eol_threshold=0.8
        )
        rul_df.to_csv('output/rul_distribution.csv', index=False)
        print("RUL分布分析已保存到 'output/rul_distribution.csv'")
    
    # 3. 特征提取
    print("\n3. 特征提取")
    extractor = BatteryFeatureExtractor(data=cleaned_data)
    
    # 提取时域特征
    if all(col in cleaned_data.columns for col in ['cycle', 'voltage', 'current', 'time', 'capacity']):
        time_features = extractor.extract_time_domain_features(
            cycle_col='cycle',
            voltage_col='voltage',
            current_col='current',
            time_col='time',
            capacity_col='capacity'
        )
        time_features.to_csv('output/time_domain_features.csv', index=False)
        print("时域特征已保存到 'output/time_domain_features.csv'")
    
    # 提取频域特征
    if all(col in cleaned_data.columns for col in ['cycle', 'voltage', 'current', 'time']):
        freq_features = extractor.extract_frequency_domain_features(
            cycle_col='cycle',
            voltage_col='voltage',
            current_col='current',
            time_col='time'
        )
        freq_features.to_csv('output/frequency_domain_features.csv', index=False)
        print("频域特征已保存到 'output/frequency_domain_features.csv'")
    
    # 提取小波特征
    if all(col in cleaned_data.columns for col in ['cycle', 'voltage', 'current', 'time']):
        wavelet_features = extractor.extract_wavelet_features(
            cycle_col='cycle',
            voltage_col='voltage',
            current_col='current',
            time_col='time'
        )
        wavelet_features.to_csv('output/wavelet_features.csv', index=False)
        print("小波特征已保存到 'output/wavelet_features.csv'")
    
    # 提取增量特征
    if 'cycle' in cleaned_data.columns:
        incremental_features = extractor.extract_incremental_features(cycle_col='cycle')
        incremental_features.to_csv('output/incremental_features.csv', index=False)
        print("增量特征已保存到 'output/incremental_features.csv'")
    
    # 提取IC曲线特征
    if all(col in cleaned_data.columns for col in ['cycle', 'voltage', 'current', 'capacity']):
        ic_features = extractor.extract_ic_curve_features(
            cycle_col='cycle',
            voltage_col='voltage',
            current_col='current',
            capacity_col='capacity'
        )
        ic_features.to_csv('output/ic_curve_features.csv', index=False)
        print("IC曲线特征已保存到 'output/ic_curve_features.csv'")
    
    # 提取充电阶段特征
    if all(col in cleaned_data.columns for col in ['cycle', 'voltage', 'current', 'time']):
        phase_features = extractor.extract_charging_phase_features(
            cycle_col='cycle',
            voltage_col='voltage',
            current_col='current',
            time_col='time'
        )
        phase_features.to_csv('output/charging_phase_features.csv', index=False)
        print("充电阶段特征已保存到 'output/charging_phase_features.csv'")
    
    # 合并所有特征
    all_features = []
    feature_files = [
        'output/time_domain_features.csv',
        'output/frequency_domain_features.csv',
        'output/wavelet_features.csv',
        'output/incremental_features.csv',
        'output/ic_curve_features.csv',
        'output/charging_phase_features.csv'
    ]
    
    for file in feature_files:
        if os.path.exists(file):
            features = pd.read_csv(file)
            if 'cycle' in features.columns:
                if not all_features:
                    all_features = features
                else:
                    # 按循环次数合并
                    all_features = pd.merge(all_features, features, on='cycle', how='inner')
    
    if all_features:
        all_features.to_csv('output/all_features.csv', index=False)
        print("所有特征已合并并保存到 'output/all_features.csv'")
    
    # 特征选择
    if all_features and 'SOH' in all_features.columns:
        # 排除cycle和SOH列
        feature_cols = [col for col in all_features.columns if col not in ['cycle', 'SOH']]
        
        if feature_cols:
            selected_features = extractor.select_features(
                features=all_features,
                feature_cols=feature_cols,
                target_col='SOH',
                method='mic',
                n_features=min(10, len(feature_cols))
            )
            selected_features.to_csv('output/selected_features.csv', index=False)
            print("选择的特征已保存到 'output/selected_features.csv'")
            
            # 特征融合
            fusion_features = extractor.feature_fusion(
                features=selected_features,
                selected_features=[col for col in selected_features.columns if col not in ['cycle', 'SOH']],
                target_col='SOH'
            )
            fusion_features.to_csv('output/fusion_features.csv', index=False)
            print("融合的特征已保存到 'output/fusion_features.csv'")
            
            # 可视化特征重要性
            extractor.visualize_feature_importance(
                features=selected_features,
                feature_cols=[col for col in selected_features.columns if col not in ['cycle', 'SOH']],
                target_col='SOH'
            )
            print("特征重要性可视化已保存")
            
            # 可视化特征趋势
            extractor.visualize_feature_trends(
                features=selected_features,
                feature_cols=[col for col in selected_features.columns if col not in ['cycle', 'SOH']],
                target_col='SOH'
            )
            print("特征趋势可视化已保存")
        else:
            print("警告：没有足够的特征列用于特征选择")
            fusion_features = all_features
    else:
        print("警告：特征数据中没有SOH列，无法进行特征选择和融合")
        fusion_features = all_features if all_features else cleaned_data
    
    # 4. 预测模型构建
    print("\n4. 预测模型构建")
    
    # 检查是否有足够的特征和目标变量
    if fusion_features is not None and 'SOH' in fusion_features.columns:
        predictor = BatteryPredictionModel(features=fusion_features)
        
        # 准备数据
        feature_cols = [col for col in fusion_features.columns if col not in ['cycle', 'SOH']]
        
        if feature_cols:
            X_train, X_test, y_train, y_test = predictor.prepare_data(
                X_cols=feature_cols,
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
                try:
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
                    print(f"{name} 模型已保存到 'models/{name.replace(' ', '_').lower()}_model.joblib'")
                except Exception as e:
                    print(f"构建和评估 {name} 模型时出错: {str(e)}")
            
            # 创建结果DataFrame
            if results:
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
                    try:
                        best_params, best_model = evaluator.optimize_best_model(
                            X_cols=feature_cols,
                            y_col='SOH',
                            model_name=best_model_name,
                            param_grid=param_grid
                        )
                        print(f"最佳参数: {best_params}")
                    except Exception as e:
                        print(f"优化 {best_model_name} 模型时出错: {str(e)}")
                
                # 交叉验证最佳模型
                print(f"\n交叉验证 {best_model_name} 模型")
                try:
                    cv_scores = evaluator.cross_validate_model(
                        X_cols=feature_cols,
                        y_col='SOH',
                        model_name=best_model_name,
                        n_splits=5
                    )
                    print(f"交叉验证分数: {cv_scores}")
                except Exception as e:
                    print(f"交叉验证 {best_model_name} 模型时出错: {str(e)}")
                
                # 绘制学习曲线
                print(f"\n绘制 {best_model_name} 学习曲线")
                try:
                    evaluator.plot_learning_curve(
                        X_cols=feature_cols,
                        y_col='SOH',
                        model_name=best_model_name
                    )
                    print(f"{best_model_name} 学习曲线已保存")
                except Exception as e:
                    print(f"绘制 {best_model_name} 学习曲线时出错: {str(e)}")
                
                # 评估集成模型
                print("\n评估集成模型")
                try:
                    top_models = results_df.head(3)['Model'].tolist()
                    ensemble_metrics = evaluator.evaluate_ensemble_model(
                        X_cols=feature_cols,
                        y_col='SOH',
                        model_names=top_models,
                        weights=[0.5, 0.3, 0.2]
                    )
                    print(f"集成模型评估结果: {ensemble_metrics}")
                except Exception as e:
                    print(f"评估集成模型时出错: {str(e)}")
                
                # 评估RUL预测
                print("\n评估RUL预测")
                try:
                    rul_metrics = evaluator.evaluate_rul_prediction(
                        X_cols=feature_cols,
                        y_col='SOH',
                        model_name=best_model_name,
                        eol_threshold=0.8
                    )
                    print(f"RUL预测评估结果: {rul_metrics}")
                except Exception as e:
                    print(f"评估RUL预测时出错: {str(e)}")
            else:
                print("警告：没有成功构建任何模型")
        else:
            print("警告：没有足够的特征列用于模型构建")
    else:
        print("警告：特征数据中没有SOH列，无法构建预测模型")
    
    print("\n电池寿命预测模型构建完成！")
    print("所有结果已保存到 'output' 目录，所有模型已保存到 'models' 目录。")
    print("请参考 'battery_prediction_guide.md' 文档了解更多详情。")


if __name__ == "__main__":
    main()
