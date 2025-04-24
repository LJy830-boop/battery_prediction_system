#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池寿命预测模型 - Web应用
该脚本实现了电池寿命预测模型的Web界面，允许用户上传数据、训练模型并可视化预测结果。
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import joblib
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing_pipeline import BatteryDataPreprocessor
from exploratory_data_analysis import BatteryDataExplorer
from feature_extraction import BatteryFeatureExtractor
from prediction_models import BatteryPredictionModel
from model_evaluation import ModelEvaluator

# 创建Flask应用
app = Flask(__name__)

# 配置上传文件存储路径
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# 确保目录存在
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制上传文件大小为100MB

# 全局变量
global_data = None
global_features = None
global_model = None
global_model_name = None
global_feature_cols = None
global_target_col = None

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def fig_to_base64(fig):
    """将matplotlib图形转换为base64编码"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_str

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传数据文件"""
    global global_data
    
    if 'file' not in request.files:
        return jsonify({'error': '没有文件部分'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        try:
            # 确保上传目录存在且有正确权限
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            os.chmod(app.config['UPLOAD_FOLDER'], 0o777)
            
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 保存文件到临时内存中
            file_content = file.read()
            
            # 写入文件
            with open(file_path, 'wb') as f:
                f.write(file_content)
            
            # 加载数据
            try:
                if filename.endswith('.csv'):
                    global_data = pd.read_csv(file_path)
                else:
                    global_data = pd.read_excel(file_path, engine='openpyxl')
                
                # 返回数据预览
                preview = global_data.head().to_html(classes='table table-striped table-bordered')
                columns = global_data.columns.tolist()
                
                return jsonify({
                    'success': True,
                    'message': f'文件 {filename} 上传成功',
                    'preview': preview,
                    'columns': columns,
                    'rows': len(global_data)
                })
            except Exception as e:
                print(f"加载数据时出错: {str(e)}")
                return jsonify({'error': f'加载数据时出错: {str(e)}'})
        except Exception as e:
            print(f"文件保存时出错: {str(e)}")
            return jsonify({'error': f'文件保存时出错: {str(e)}'})
    
    return jsonify({'error': '不允许的文件类型'})

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    """预处理数据"""
    global global_data
    
    if global_data is None:
        return jsonify({'error': '请先上传数据'})
    
    try:
        # 创建预处理器
        preprocessor = BatteryDataPreprocessor(data=global_data)
        
        # 数据清洗
        drop_na = request.form.get('drop_na') == 'true'
        drop_duplicates = request.form.get('drop_duplicates') == 'true'
        
        cleaned_data = preprocessor.clean_data(drop_na=drop_na, drop_duplicates=drop_duplicates)
        global_data = cleaned_data
        
        # 返回清洗后的数据预览
        preview = global_data.head().to_html(classes='table table-striped table-bordered')
        
        return jsonify({
            'success': True,
            'message': '数据预处理完成',
            'preview': preview,
            'rows': len(global_data)
        })
    except Exception as e:
        return jsonify({'error': f'预处理数据时出错: {str(e)}'})

@app.route('/analyze', methods=['POST'])
def analyze_data():
    """探索性数据分析"""
    global global_data
    
    if global_data is None:
        return jsonify({'error': '请先上传并预处理数据'})
    
    try:
        # 创建数据探索器
        explorer = BatteryDataExplorer(data=global_data)
        
        # 数据摘要
        summary = explorer.data_summary()
        summary_html = summary.to_html(classes='table table-striped table-bordered')
        
        # 数据分布可视化
        plt.figure(figsize=(12, 6))
        for i, col in enumerate(global_data.select_dtypes(include=[np.number]).columns[:5]):
            plt.subplot(1, 5, i+1)
            sns.histplot(global_data[col], kde=True)
            plt.title(col)
            plt.tight_layout()
        dist_plot = fig_to_base64(plt.gcf())
        plt.close()
        
        # 相关性矩阵
        plt.figure(figsize=(10, 8))
        corr_matrix = global_data.select_dtypes(include=[np.number]).corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('相关性矩阵')
        corr_plot = fig_to_base64(plt.gcf())
        plt.close()
        
        # 容量退化曲线（如果有相关列）
        capacity_plot = None
        if all(col in global_data.columns for col in ['cycle', 'capacity']):
            plt.figure(figsize=(10, 6))
            explorer.visualize_capacity_degradation(cycle_col='cycle', capacity_col='capacity')
            capacity_plot = fig_to_base64(plt.gcf())
            plt.close()
        
        return jsonify({
            'success': True,
            'message': '探索性数据分析完成',
            'summary': summary_html,
            'dist_plot': dist_plot,
            'corr_plot': corr_plot,
            'capacity_plot': capacity_plot
        })
    except Exception as e:
        return jsonify({'error': f'分析数据时出错: {str(e)}'})

@app.route('/extract_features', methods=['POST'])
def extract_features():
    """特征提取"""
    global global_data, global_features, global_feature_cols, global_target_col
    
    if global_data is None:
        return jsonify({'error': '请先上传并预处理数据'})
    
    try:
        # 获取表单数据
        cycle_col = request.form.get('cycle_col')
        voltage_col = request.form.get('voltage_col')
        current_col = request.form.get('current_col')
        time_col = request.form.get('time_col')
        capacity_col = request.form.get('capacity_col')
        target_col = request.form.get('target_col')
        
        # 保存目标列
        global_target_col = target_col
        
        # 创建特征提取器
        extractor = BatteryFeatureExtractor(data=global_data)
        
        # 提取特征
        features_list = []
        feature_names = []
        
        # 提取时域特征
        if all(col in global_data.columns for col in [cycle_col, voltage_col, current_col, time_col, capacity_col]):
            time_features = extractor.extract_time_domain_features(
                cycle_col=cycle_col,
                voltage_col=voltage_col,
                current_col=current_col,
                time_col=time_col,
                capacity_col=capacity_col
            )
            features_list.append(time_features)
            feature_names.append('时域特征')
        
        # 提取频域特征
        if all(col in global_data.columns for col in [cycle_col, voltage_col, current_col, time_col]):
            freq_features = extractor.extract_frequency_domain_features(
                cycle_col=cycle_col,
                voltage_col=voltage_col,
                current_col=current_col,
                time_col=time_col
            )
            features_list.append(freq_features)
            feature_names.append('频域特征')
        
        # 提取增量特征
        if cycle_col in global_data.columns:
            incremental_features = extractor.extract_incremental_features(cycle_col=cycle_col)
            features_list.append(incremental_features)
            feature_names.append('增量特征')
        
        # 提取IC曲线特征
        if all(col in global_data.columns for col in [cycle_col, voltage_col, current_col, capacity_col]):
            ic_features = extractor.extract_ic_curve_features(
                cycle_col=cycle_col,
                voltage_col=voltage_col,
                current_col=current_col,
                capacity_col=capacity_col
            )
            features_list.append(ic_features)
            feature_names.append('IC曲线特征')
        
        # 合并所有特征
        all_features = None
        for features in features_list:
            if all_features is None:
                all_features = features
            else:
                # 按循环次数合并
                all_features = pd.merge(all_features, features, on=cycle_col, how='inner')
        
        # 如果没有提取到特征，使用原始数据
        if all_features is None:
            all_features = global_data
        
        # 特征选择
        if target_col in all_features.columns:
            # 排除cycle和target_col列
            feature_cols = [col for col in all_features.columns if col not in [cycle_col, target_col]]
            
            if feature_cols:
                selected_features = extractor.select_features(
                    features=all_features,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    method='mic',
                    n_features=min(10, len(feature_cols))
                )
                
                # 保存特征列
                global_feature_cols = [col for col in selected_features.columns if col not in [cycle_col, target_col]]
                
                # 保存特征数据
                global_features = selected_features
                
                # 特征重要性可视化
                plt.figure(figsize=(10, 6))
                extractor.visualize_feature_importance(
                    features=selected_features,
                    feature_cols=global_feature_cols,
                    target_col=target_col
                )
                importance_plot = fig_to_base64(plt.gcf())
                plt.close()
                
                # 特征趋势可视化
                plt.figure(figsize=(12, 8))
                extractor.visualize_feature_trends(
                    features=selected_features,
                    feature_cols=global_feature_cols[:5],  # 只显示前5个特征
                    target_col=target_col
                )
                trends_plot = fig_to_base64(plt.gcf())
                plt.close()
                
                # 返回特征预览
                preview = selected_features.head().to_html(classes='table table-striped table-bordered')
                
                return jsonify({
                    'success': True,
                    'message': '特征提取完成',
                    'preview': preview,
                    'importance_plot': importance_plot,
                    'trends_plot': trends_plot,
                    'feature_names': feature_names,
                    'feature_cols': global_feature_cols,
                    'rows': len(selected_features)
                })
            else:
                return jsonify({'error': '没有足够的特征列用于特征选择'})
        else:
            return jsonify({'error': f'目标列 {target_col} 不在特征数据中'})
    except Exception as e:
        return jsonify({'error': f'提取特征时出错: {str(e)}'})

@app.route('/train_model', methods=['POST'])
def train_model():
    """训练模型"""
    global global_features, global_model, global_model_name, global_feature_cols, global_target_col
    
    if global_features is None:
        return jsonify({'error': '请先提取特征'})
    
    try:
        # 获取表单数据
        model_type = request.form.get('model_type')
        test_size = float(request.form.get('test_size', 0.2))
        
        # 创建预测模型实例
        predictor = BatteryPredictionModel(features=global_features)
        
        # 准备数据
        X_train, X_test, y_train, y_test = predictor.prepare_data(
            X_cols=global_feature_cols,
            y_col=global_target_col,
            test_size=test_size
        )
        
        # 构建模型
        if model_type == 'svr':
            predictor.build_svr_model()
            global_model_name = 'SVR'
        elif model_type == 'rf':
            predictor.build_random_forest_model()
            global_model_name = 'Random Forest'
        elif model_type == 'gb':
            predictor.build_gradient_boosting_model()
            global_model_name = 'Gradient Boosting'
        elif model_type == 'xgb':
            predictor.build_xgboost_model()
            global_model_name = 'XGBoost'
        elif model_type == 'lgb':
            predictor.build_lightgbm_model()
            global_model_name = 'LightGBM'
        elif model_type == 'linear':
            predictor.build_linear_model(model_type='linear')
            global_model_name = 'Linear Regression'
        elif model_type == 'mlp':
            predictor.build_mlp_model(input_dim=X_train.shape[1])
            global_model_name = 'MLP'
        else:
            return jsonify({'error': f'不支持的模型类型: {model_type}'})
        
        # 训练模型
        predictor.train_model(X_train, y_train)
        
        # 评估模型
        metrics = predictor.evaluate_model(X_test, y_test, plot=True)
        
        # 保存模型
        model_path = os.path.join(MODELS_FOLDER, f'{model_type}_model.joblib')
        predictor.save_model(model_path)
        
        # 保存模型实例
        global_model = predictor
        
        # 获取预测结果图
        prediction_plot = fig_to_base64(plt.gcf())
        plt.close()
        
        # 如果是基于树的模型，可视化特征重要性
        importance_plot = None
        if model_type in ['rf', 'gb', 'xgb', 'lgb']:
            plt.figure(figsize=(10, 6))
            importance_df = predictor.visualize_feature_importance(global_feature_cols)
            importance_plot = fig_to_base64(plt.gcf())
            plt.close()
        
        return jsonify({
            'success': True,
            'message': f'{global_model_name} 模型训练完成',
            'metrics': {
                'mse': metrics['mse'],
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2']
            },
            'prediction_plot': prediction_plot,
            'importance_plot': importance_plot
        })
    except Exception as e:
        return jsonify({'error': f'训练模型时出错: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict():
    """预测SOH和RUL"""
    global global_model, global_features, global_feature_cols, global_target_col
    
    if global_model is None:
        return jsonify({'error': '请先训练模型'})
    
    try:
        # 获取表单数据
        cycle = int(request.form.get('cycle'))
        current_soh = float(request.form.get('current_soh', 0.9))
        eol_threshold = float(request.form.get('eol_threshold', 0.8))
        
        # 获取指定循环的特征
        if 'cycle' in global_features.columns:
            cycle_data = global_features[global_features['cycle'] == cycle]
            
            if len(cycle_data) == 0:
                return jsonify({'error': f'找不到循环 {cycle} 的数据'})
            
            # 提取特征
            X = cycle_data[global_feature_cols].values.reshape(1, -1)
            
            # 预测SOH
            soh_pred = global_model.predict_soh(X)[0]
            
            # 预测RUL
            rul_pred = global_model.predict_rul(X, current_soh=soh_pred, eol_threshold=eol_threshold)
            
            # 可视化RUL预测
            plt.figure(figsize=(8, 6))
            global_model.visualize_rul_prediction(X, current_soh=soh_pred, eol_threshold=eol_threshold)
            rul_plot = fig_to_base64(plt.gcf())
            plt.close()
            
            # 可视化SOH随循环次数的变化
            plt.figure(figsize=(10, 6))
            cycles = global_features['cycle'].values
            actual_soh = global_features[global_target_col].values
            
            # 预测所有循环的SOH
            X_all = global_features[global_feature_cols].values
            pred_soh = global_model.predict_soh(X_all)
            
            plt.plot(cycles, actual_soh, 'b-', label='实际SOH')
            plt.plot(cycles, pred_soh, 'r--', label='预测SOH')
            plt.axvline(x=cycle, color='g', linestyle='--', label=f'当前循环 {cycle}')
            plt.xlabel('循环次数')
            plt.ylabel('SOH')
            plt.title('SOH随循环次数的变化')
            plt.legend()
            plt.grid(True)
            soh_plot = fig_to_base64(plt.gcf())
            plt.close()
            
            return jsonify({
                'success': True,
                'message': '预测完成',
                'soh_pred': soh_pred,
                'rul_pred': rul_pred,
                'rul_plot': rul_plot,
                'soh_plot': soh_plot
            })
        else:
            return jsonify({'error': '特征数据中没有cycle列'})
    except Exception as e:
        return jsonify({'error': f'预测时出错: {str(e)}'})

@app.route('/optimize_model', methods=['POST'])
def optimize_model():
    """优化模型"""
    global global_features, global_model, global_model_name, global_feature_cols, global_target_col
    
    if global_features is None:
        return jsonify({'error': '请先提取特征'})
    
    try:
        # 获取表单数据
        model_type = request.form.get('model_type')
        
        # 创建模型评估器
        evaluator = ModelEvaluator(features=global_features)
        
        # 根据模型类型定义参数网格
        if model_type == 'svr':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.1, 0.01],
                'kernel': ['rbf', 'linear', 'poly']
            }
            model_name = 'SVR'
        elif model_type == 'rf':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None]
            }
            model_name = 'Random Forest'
        elif model_type == 'gb':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model_name = 'Gradient Boosting'
        elif model_type == 'xgb':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model_name = 'XGBoost'
        elif model_type == 'lgb':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            model_name = 'LightGBM'
        elif model_type == 'linear':
            param_grid = {}
            model_name = 'Linear Regression'
        else:
            return jsonify({'error': f'不支持的模型类型: {model_type}'})
        
        # 优化模型
        if param_grid:
            best_params, best_model = evaluator.optimize_best_model(
                X_cols=global_feature_cols,
                y_col=global_target_col,
                model_name=model_name,
                param_grid=param_grid
            )
            
            # 保存最佳模型
            model_path = os.path.join(MODELS_FOLDER, f'{model_type}_optimized_model.joblib')
            joblib.dump(best_model, model_path)
            
            # 更新全局模型
            predictor = BatteryPredictionModel(features=global_features)
            predictor.model = best_model
            predictor.scaler_X = evaluator.scaler_X
            predictor.scaler_y = evaluator.scaler_y
            global_model = predictor
            global_model_name = model_name
            
            # 交叉验证
            cv_scores = evaluator.cross_validate_model(
                X_cols=global_feature_cols,
                y_col=global_target_col,
                model_name=model_name,
                n_splits=5
            )
            
            # 获取交叉验证图
            cv_plot = fig_to_base64(plt.gcf())
            plt.close()
            
            # 绘制学习曲线
            evaluator.plot_learning_curve(
                X_cols=global_feature_cols,
                y_col=global_target_col,
                model_name=model_name
            )
            
            # 获取学习曲线图
            learning_curve_plot = fig_to_base64(plt.gcf())
            plt.close()
            
            return jsonify({
                'success': True,
                'message': f'{model_name} 模型优化完成',
                'best_params': best_params,
                'cv_scores': {
                    'mse': cv_scores['mse'],
                    'rmse': cv_scores['rmse'],
                    'mae': cv_scores['mae'],
                    'r2': cv_scores['r2']
                },
                'cv_plot': cv_plot,
                'learning_curve_plot': learning_curve_plot
            })
        else:
            return jsonify({'error': '没有可优化的参数'})
    except Exception as e:
        return jsonify({'error': f'优化模型时出错: {str(e)}'})

@app.route('/ensemble_model', methods=['POST'])
def ensemble_model():
    """集成模型"""
    global global_features, global_feature_cols, global_target_col
    
    if global_features is None:
        return jsonify({'error': '请先提取特征'})
    
    try:
        # 获取表单数据
        model_types = request.form.getlist('model_types[]')
        weights = request.form.getlist('weights[]')
        
        # 转换权重为浮点数
        weights = [float(w) for w in weights]
        
        # 确保权重和模型数量相同
        if len(weights) != len(model_types):
            return jsonify({'error': '权重数量必须与模型数量相同'})
        
        # 创建模型评估器
        evaluator = ModelEvaluator(features=global_features)
        
        # 将模型类型转换为模型名称
        model_names = []
        for model_type in model_types:
            if model_type == 'svr':
                model_names.append('SVR')
            elif model_type == 'rf':
                model_names.append('Random Forest')
            elif model_type == 'gb':
                model_names.append('Gradient Boosting')
            elif model_type == 'xgb':
                model_names.append('XGBoost')
            elif model_type == 'lgb':
                model_names.append('LightGBM')
            elif model_type == 'linear':
                model_names.append('Linear Regression')
            else:
                return jsonify({'error': f'不支持的模型类型: {model_type}'})
        
        # 评估集成模型
        ensemble_metrics = evaluator.evaluate_ensemble_model(
            X_cols=global_feature_cols,
            y_col=global_target_col,
            model_names=model_names,
            weights=weights
        )
        
        # 获取集成模型预测图
        ensemble_plot = fig_to_base64(plt.gcf())
        plt.close()
        
        return jsonify({
            'success': True,
            'message': '集成模型评估完成',
            'metrics': {
                'mse': ensemble_metrics['mse'],
                'rmse': ensemble_metrics['rmse'],
                'mae': ensemble_metrics['mae'],
                'r2': ensemble_metrics['r2']
            },
            'ensemble_plot': ensemble_plot
        })
    except Exception as e:
        return jsonify({'error': f'评估集成模型时出错: {str(e)}'})

@app.route('/evaluate_rul', methods=['POST'])
def evaluate_rul():
    """评估RUL预测"""
    global global_features, global_feature_cols, global_target_col
    
    if global_features is None:
        return jsonify({'error': '请先提取特征'})
    
    try:
        # 获取表单数据
        model_type = request.form.get('model_type')
        eol_threshold = float(request.form.get('eol_threshold', 0.8))
        
        # 将模型类型转换为模型名称
        if model_type == 'svr':
            model_name = 'SVR'
        elif model_type == 'rf':
            model_name = 'Random Forest'
        elif model_type == 'gb':
            model_name = 'Gradient Boosting'
        elif model_type == 'xgb':
            model_name = 'XGBoost'
        elif model_type == 'lgb':
            model_name = 'LightGBM'
        elif model_type == 'linear':
            model_name = 'Linear Regression'
        else:
            return jsonify({'error': f'不支持的模型类型: {model_type}'})
        
        # 创建模型评估器
        evaluator = ModelEvaluator(features=global_features)
        
        # 评估RUL预测
        rul_metrics = evaluator.evaluate_rul_prediction(
            X_cols=global_feature_cols,
            y_col=global_target_col,
            model_name=model_name,
            eol_threshold=eol_threshold
        )
        
        # 获取RUL预测图
        rul_plot = fig_to_base64(plt.gcf())
        plt.close()
        
        # 获取RUL随循环次数的变化图
        rul_vs_cycle_plot = fig_to_base64(plt.gcf())
        plt.close()
        
        return jsonify({
            'success': True,
            'message': 'RUL预测评估完成',
            'metrics': {
                'mse': rul_metrics['mse'],
                'rmse': rul_metrics['rmse'],
                'mae': rul_metrics['mae']
            },
            'rul_plot': rul_plot,
            'rul_vs_cycle_plot': rul_vs_cycle_plot
        })
    except Exception as e:
        return jsonify({'error': f'评估RUL预测时出错: {str(e)}'})

@app.route('/download_model', methods=['GET'])
def download_model():
    """下载模型"""
    model_type = request.args.get('model_type')
    optimized = request.args.get('optimized') == 'true'
    
    if optimized:
        model_path = os.path.join(MODELS_FOLDER, f'{model_type}_optimized_model.joblib')
    else:
        model_path = os.path.join(MODELS_FOLDER, f'{model_type}_model.joblib')
    
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({'error': '模型文件不存在'})

@app.route('/download_features', methods=['GET'])
def download_features():
    """下载特征数据"""
    global global_features
    
    if global_features is None:
        return jsonify({'error': '请先提取特征'})
    
    # 保存特征数据
    features_path = os.path.join(OUTPUT_FOLDER, 'features.csv')
    global_features.to_csv(features_path, index=False)
    
    return send_file(features_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
