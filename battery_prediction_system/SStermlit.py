# coding: utf-8
"""
电池寿命预测模型 - Streamlit应用
该脚本实现了电池寿命预测模型的Streamlit界面，允许用户上传数据、训练模型并可视化预测结果。
添加了服务器连接选项，支持本地和远程两种模式。
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import warnings
import requests
import json
import time
from urllib.parse import urljoin

warnings.filterwarnings('ignore')

# 导入自定义模块
from data_preprocessing_pipeline import BatteryDataPreprocessor
from exploratory_data_analysis import BatteryDataExplorer
from feature_extraction import BatteryFeatureExtractor
from prediction_models import BatteryPredictionModel
from model_evaluation import ModelEvaluator

# 配置页面
st.set_page_config(
    page_title="电池寿命预测系统",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 配置上传文件存储路径
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MODELS_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# 确保目录存在
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODELS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# 初始化会话状态
if 'data' not in st.session_state:
    st.session_state.data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'server_mode' not in st.session_state:
    st.session_state.server_mode = False
if 'server_url' not in st.session_state:
    st.session_state.server_url = ""
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# 辅助函数
def allowed_file(filename):
    """检查文件类型是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_image_base64(fig):
    """将matplotlib图形转换为base64编码"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

def test_server_connection(url, api_key=None):
    """测试服务器连接"""
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        response = requests.get(urljoin(url, "/ping"), headers=headers, timeout=5)
        if response.status_code == 200 and response.json().get("status") == "ok":
            return True
    except Exception:
        pass
    return False

def upload_file_to_server(url, file, api_key=None):
    """上传文件到服务器"""
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        files = {'file': (file.name, file.getvalue())}
        response = requests.post(urljoin(url, "/upload"), files=files, headers=headers, timeout=30)
        
        if response.status_code == 200:
            return response.json().get("file_id")
    except Exception as e:
        st.error(f"文件上传失败: {str(e)}")
    return None

def send_request_to_server(url, endpoint, data, api_key=None):
    """发送请求到服务器"""
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
            
        response = requests.post(
            urljoin(url, endpoint), 
            json=data, 
            headers=headers, 
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"服务器请求失败: {str(e)}")
    return None

# 侧边栏导航
st.sidebar.title("电池寿命预测系统")
st.sidebar.image("https://img.icons8.com/color/96/000000/battery-level.png", width=100)

# 添加服务器连接选项
st.sidebar.markdown("---")
st.sidebar.markdown("### 服务器连接配置")
server_mode = st.sidebar.checkbox("启用服务器模式", value=st.session_state.server_mode)

if server_mode:
    st.session_state.server_mode = True
    server_url = st.sidebar.text_input(
        "服务器地址", 
        value=st.session_state.server_url or "http://localhost:8000",
        help="请输入服务器地址，格式如：http://your-server.com:8000"
    )
    api_key = st.sidebar.text_input(
        "API密钥 (可选)", 
        value=st.session_state.api_key or "",
        type="password",
        help="如果需要身份验证，请输入API密钥"
    )
    
    st.session_state.server_url = server_url
    st.session_state.api_key = api_key
    
    if st.sidebar.button("测试服务器连接"):
        if test_server_connection(server_url, api_key):
            st.sidebar.success("服务器连接成功！")
        else:
            st.sidebar.error("无法连接到服务器，请检查地址和密钥")
else:
    st.session_state.server_mode = False

step = st.sidebar.radio(
    "导航",
    ["1. 数据上传", "2. 数据预处理", "3. 探索性分析", "4. 特征提取", 
     "5. 模型训练", "6. 预测与评估", "7. 模型优化"],
    index=st.session_state.current_step - 1
)

st.session_state.current_step = int(step[0])

# 1. 数据上传页面
if st.session_state.current_step == 1:
    st.title("1. 数据上传")
    st.write("上传电池数据文件（支持CSV和Excel格式）")
    
    uploaded_file = st.file_uploader("选择数据文件", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # 在服务器模式下，上传文件到服务器
            if st.session_state.server_mode:
                with st.spinner("正在上传文件到服务器..."):
                    file_id = upload_file_to_server(
                        st.session_state.server_url, 
                        uploaded_file, 
                        st.session_state.api_key
                    )
                    
                if file_id:
                    st.success(f"文件 {uploaded_file.name} 已成功上传到服务器！文件ID: {file_id}")
                    st.session_state.file_id = file_id
                else:
                    st.error("文件上传失败，请检查服务器连接")
                    return
            else:
                # 保存上传的文件
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 加载数据
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(file_path)
                else:
                    st.session_state.data = pd.read_excel(file_path, engine='openpyxl')
                
                st.success(f"文件 {uploaded_file.name} 上传成功！")
                
                # 显示数据预览
                st.subheader("数据预览")
                st.dataframe(st.session_state.data.head())
                
                st.info(f"数据形状: {st.session_state.data.shape[0]} 行, {st.session_state.data.shape[1]} 列")
                
                # 显示列信息
                st.subheader("列信息")
                col_info = pd.DataFrame({
                    '列名': st.session_state.data.columns,
                    '数据类型': st.session_state.data.dtypes.astype(str),
                    '非空值数量': st.session_state.data.count().values,
                    '空值数量': st.session_state.data.isna().sum().values,
                    '唯一值数量': [st.session_state.data[col].nunique() for col in st.session_state.data.columns]
                })
                st.dataframe(col_info)
                
            # 下一步按钮
            if st.button("继续到数据预处理"):
                st.session_state.current_step = 2
                st.rerun()
                
        except Exception as e:
            st.error(f"加载数据时出错: {str(e)}")

# 2. 数据预处理页面
elif st.session_state.current_step == 2:
    st.title("2. 数据预处理")
    
    if st.session_state.server_mode and not hasattr(st.session_state, 'file_id'):
        st.warning("请先上传数据文件")
        if st.button("返回数据上传"):
            st.session_state.current_step = 1
            st.rerun()
    elif not st.session_state.server_mode and st.session_state.data is None:
        st.warning("请先上传数据文件")
        if st.button("返回数据上传"):
            st.session_state.current_step = 1
            st.rerun()
    else:
        st.write("选择数据预处理选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("选择列")
            cycle_col = st.selectbox("循环次数列", st.session_state.data.columns if not st.session_state.server_mode else [])
            voltage_col = st.selectbox("电压列", st.session_state.data.columns if not st.session_state.server_mode else [])
            current_col = st.selectbox("电流列", st.session_state.data.columns if not st.session_state.server_mode else [])
            time_col = st.selectbox("时间列", st.session_state.data.columns if not st.session_state.server_mode else [])
            
            capacity_col = st.selectbox(
                "容量列 (可选)", 
                ["无"] + (list(st.session_state.data.columns) if not st.session_state.server_mode else [])
            )
            capacity_col = None if capacity_col == "无" else capacity_col
            
            temp_col = st.selectbox(
                "温度列 (可选)", 
                ["无"] + (list(st.session_state.data.columns) if not st.session_state.server_mode else [])
            )
            temp_col = None if temp_col == "无" else temp_col
        
        with col2:
            st.subheader("预处理选项")
            remove_outliers = st.checkbox("移除异常值", value=True)
            fill_missing = st.checkbox("填充缺失值", value=True)
            normalize_data = st.checkbox("标准化数据", value=True)
            
            outlier_threshold = st.slider(
                "异常值阈值 (标准差倍数)", 
                min_value=1.0, 
                max_value=5.0, 
                value=3.0, 
                step=0.1
            )
        
        if st.button("执行数据预处理"):
            try:
                if st.session_state.server_mode:
                    with st.spinner("正在服务器上预处理数据..."):
                        request_data = {
                            "file_id": st.session_state.file_id,
                            "options": {
                                "cycle_col": cycle_col,
                                "voltage_col": voltage_col,
                                "current_col": current_col,
                                "time_col": time_col,
                                "capacity_col": capacity_col,
                                "temp_col": temp_col,
                                "remove_outliers": remove_outliers,
                                "fill_missing": fill_missing,
                                "normalize": normalize_data,
                                "outlier_threshold": outlier_threshold
                            }
                        }
                        
                        response = send_request_to_server(
                            st.session_state.server_url,
                            "/preprocess",
                            request_data,
                            st.session_state.api_key
                        )
                        
                        if response and response.get("status") == "success":
                            st.success("服务器数据预处理完成！")
                            st.session_state.preprocessed_file_id = response.get("file_id")
                            
                            # 显示预处理统计信息
                            st.subheader("预处理统计信息")
                            stats = response.get("stats", {})
                            st.json(stats)
                            
                            # 提供下载链接
                            download_url = urljoin(
                                st.session_state.server_url, 
                                f"/download/{st.session_state.preprocessed_file_id}"
                            )
                            st.markdown(f"[下载预处理后的数据]({download_url})")
                        else:
                            st.error("服务器预处理失败")
                else:
                    with st.spinner("正在预处理数据..."):
                        # 创建预处理器
                        preprocessor = BatteryDataPreprocessor(st.session_state.data)
                        
                        # 执行预处理
                        preprocessor.preprocess_data(
                            cycle_col=cycle_col,
                            voltage_col=voltage_col,
                            current_col=current_col,
                            time_col=time_col,
                            capacity_col=capacity_col,
                            temp_col=temp_col,
                            remove_outliers=remove_outliers,
                            fill_missing=fill_missing,
                            normalize=normalize_data,
                            outlier_threshold=outlier_threshold
                        )
                        
                        # 更新会话状态
                        st.session_state.data = preprocessor.processed_data
                        
                        # 显示预处理结果
                        st.success("数据预处理完成！")
                        st.subheader("预处理后的数据")
                        st.dataframe(st.session_state.data.head())
                        
                        # 显示预处理统计信息
                        st.subheader("预处理统计信息")
                        stats = {
                            "原始数据行数": preprocessor.original_data.shape[0],
                            "预处理后行数": preprocessor.processed_data.shape[0],
                            "移除的异常值数": preprocessor.original_data.shape[0] - preprocessor.processed_data.shape[0] if remove_outliers else 0,
                            "填充的缺失值数": preprocessor.missing_values_filled if fill_missing else 0
                        }
                        st.json(stats)
                        
                        # 保存预处理后的数据
                        preprocessed_file = os.path.join(OUTPUT_FOLDER, "preprocessed_data.csv")
                        st.session_state.data.to_csv(preprocessed_file, index=False)
                        
                        # 提供下载链接
                        with open(preprocessed_file, "rb") as file:
                            st.download_button(
                                label="下载预处理后的数据",
                                data=file,
                                file_name="preprocessed_data.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"预处理数据时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回数据上传"):
                st.session_state.current_step = 1
                st.rerun()
        with col2:
            if st.button("继续到探索性分析"):
                st.session_state.current_step = 3
                st.rerun()

# 3. 探索性分析页面
elif st.session_state.current_step == 3:
    st.title("3. 探索性数据分析")
    
    if st.session_state.server_mode and not hasattr(st.session_state, 'preprocessed_file_id'):
        st.warning("请先预处理数据")
        if st.button("返回数据预处理"):
            st.session_state.current_step = 2
            st.rerun()
    elif not st.session_state.server_mode and st.session_state.data is None:
        st.warning("请先预处理数据")
        if st.button("返回数据预处理"):
            st.session_state.current_step = 2
            st.rerun()
    else:
        st.write("选择探索性分析选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("选择列")
            cycle_col = st.selectbox("循环次数列", st.session_state.data.columns if not st.session_state.server_mode else [])
            voltage_col = st.selectbox("电压列", st.session_state.data.columns if not st.session_state.server_mode else [])
            current_col = st.selectbox("电流列", st.session_state.data.columns if not st.session_state.server_mode else [])
            
            capacity_col = st.selectbox(
                "容量列 (可选)", 
                ["无"] + (list(st.session_state.data.columns) if not st.session_state.server_mode else [])
            )
            capacity_col = None if capacity_col == "无" else capacity_col
        
        with col2:
            st.subheader("分析选项")
            show_summary = st.checkbox("显示数据摘要", value=True)
            show_distributions = st.checkbox("显示分布图", value=True)
            show_correlations = st.checkbox("显示相关性矩阵", value=True)
            show_capacity_fade = st.checkbox("显示容量退化曲线", value=True)
        
        if st.button("执行探索性分析"):
            try:
                if st.session_state.server_mode:
                    with st.spinner("正在服务器上进行分析..."):
                        request_data = {
                            "file_id": st.session_state.preprocessed_file_id,
                            "options": {
                                "cycle_col": cycle_col,
                                "voltage_col": voltage_col,
                                "current_col": current_col,
                                "capacity_col": capacity_col,
                                "show_summary": show_summary,
                                "show_distributions": show_distributions,
                                "show_correlations": show_correlations,
                                "show_capacity_fade": show_capacity_fade
                            }
                        }
                        
                        response = send_request_to_server(
                            st.session_state.server_url,
                            "/explore",
                            request_data,
                            st.session_state.api_key
                        )
                        
                        if response and response.get("status") == "success":
                            st.success("探索性数据分析完成！")
                            
                            # 显示分析结果
                            if show_summary and "summary" in response:
                                st.subheader("数据摘要")
                                st.json(response["summary"])
                            
                            if show_distributions and "distributions" in response:
                                st.subheader("数据分布")
                                # 假设服务器返回的是base64编码的图像
                                for img_base64 in response["distributions"]:
                                    st.image(f"data:image/png;base64,{img_base64}")
                            
                            if show_correlations and "correlation" in response:
                                st.subheader("相关性矩阵")
                                st.image(f"data:image/png;base64,{response['correlation']}")
                            
                            if show_capacity_fade and capacity_col and "capacity_fade" in response:
                                st.subheader("容量退化曲线")
                                st.image(f"data:image/png;base64,{response['capacity_fade']}")
                                
                                st.subheader("健康状态 (SOH) 曲线")
                                st.image(f"data:image/png;base64,{response['soh_curve']}")
                            
                            if "voltage_current" in response:
                                st.subheader("电压-电流关系")
                                st.image(f"data:image/png;base64,{response['voltage_current']}")
                        else:
                            st.error("服务器分析失败")
                else:
                    with st.spinner("正在分析数据..."):
                        # 创建数据探索器
                        explorer = BatteryDataExplorer(st.session_state.data)
                        
                        # 数据摘要
                        if show_summary:
                            st.subheader("数据摘要")
                            st.dataframe(st.session_state.data.describe())
                        
                        # 分布图
                        if show_distributions:
                            st.subheader("数据分布")
                            
                            # 选择要显示的列
                            cols_to_plot = st.multiselect(
                                "选择要显示分布的列",
                                st.session_state.data.select_dtypes(include=np.number).columns.tolist(),
                                default=[voltage_col, current_col]
                            )
                            
                            if cols_to_plot:
                                fig = explorer.plot_distributions(cols_to_plot)
                                st.pyplot(fig)
                        
                        # 相关性矩阵
                        if show_correlations:
                            st.subheader("相关性矩阵")
                            fig = explorer.plot_correlation_matrix()
                            st.pyplot(fig)
                        
                        # 容量退化曲线
                        if show_capacity_fade and capacity_col:
                            st.subheader("容量退化曲线")
                            fig = explorer.plot_capacity_fade(cycle_col, capacity_col)
                            st.pyplot(fig)
                            
                            # 计算SOH
                            st.subheader("健康状态 (SOH) 曲线")
                            fig = explorer.plot_soh_curve(cycle_col, capacity_col)
                            st.pyplot(fig)
                        
                        # 电压-电流关系
                        st.subheader("电压-电流关系")
                        fig = explorer.plot_voltage_current_relationship(voltage_col, current_col, cycle_col)
                        st.pyplot(fig)
                        
                        # 保存分析结果
                        output_file = os.path.join(OUTPUT_FOLDER, "eda_results.png")
                        fig.savefig(output_file, bbox_inches='tight')
                        
                        st.success("探索性数据分析完成！")
            
            except Exception as e:
                st.error(f"分析数据时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回数据预处理"):
                st.session_state.current_step = 2
                st.rerun()
        with col2:
            if st.button("继续到特征提取"):
                st.session_state.current_step = 4
                st.rerun()

# 4. 特征提取页面
elif st.session_state.current_step == 4:
    st.title("4. 特征提取")
    
    if st.session_state.server_mode and not hasattr(st.session_state, 'preprocessed_file_id'):
        st.warning("请先预处理数据")
        if st.button("返回探索性分析"):
            st.session_state.current_step = 3
            st.rerun()
    elif not st.session_state.server_mode and st.session_state.data is None:
        st.warning("请先预处理数据")
        if st.button("返回探索性分析"):
            st.session_state.current_step = 3
            st.rerun()
    else:
        st.write("选择特征提取选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("选择列")
            cycle_col = st.selectbox("循环次数列", st.session_state.data.columns if not st.session_state.server_mode else [])
            voltage_col = st.selectbox("电压列", st.session_state.data.columns if not st.session_state.server_mode else [])
            current_col = st.selectbox("电流列", st.session_state.data.columns if not st.session_state.server_mode else [])
            time_col = st.selectbox("时间列", st.session_state.data.columns if not st.session_state.server_mode else [])
            
            capacity_col = st.selectbox(
                "容量列 (可选)", 
                ["无"] + (list(st.session_state.data.columns) if not st.session_state.server_mode else [])
            )
            capacity_col = None if capacity_col == "无" else capacity_col
        
        with col2:
            st.subheader("特征提取选项")
            extract_time_domain = st.checkbox("提取时域特征", value=True)
            extract_frequency_domain = st.checkbox("提取频域特征", value=True)
            extract_wavelet = st.checkbox("提取小波特征", value=True)
            extract_incremental = st.checkbox("提取增量特征", value=True)
            extract_ic_curve = st.checkbox("提取IC曲线特征", value=True)
        
        if st.button("执行特征提取"):
            try:
                if st.session_state.server_mode:
                    with st.spinner("正在服务器上提取特征..."):
                        request_data = {
                            "file_id": st.session_state.preprocessed_file_id,
                            "options": {
                                "cycle_col": cycle_col,
                                "voltage_col": voltage_col,
                                "current_col": current_col,
                                "time_col": time_col,
                                "capacity_col": capacity_col,
                                "extract_time_domain": extract_time_domain,
                                "extract_frequency_domain": extract_frequency_domain,
                                "extract_wavelet": extract_wavelet,
                                "extract_incremental": extract_incremental,
                                "extract_ic_curve": extract_ic_curve
                            }
                        }
                        
                        response = send_request_to_server(
                            st.session_state.server_url,
                            "/extract_features",
                            request_data,
                            st.session_state.api_key
                        )
                        
                        if response and response.get("status") == "success":
                            st.success("特征提取完成！")
                            
                            # 下载特征文件
                            features_file_id = response.get("file_id")
                            download_url = urljoin(
                                st.session_state.server_url, 
                                f"/download/{features_file_id}"
                            )
                            st.markdown(f"[下载提取的特征]({download_url})")
                            
                            # 显示特征统计信息
                            st.subheader("特征统计信息")
                            st.info(f"共提取了 {response.get('feature_count', 0)} 个特征，覆盖 {response.get('cycle_count', 0)} 个循环")
                            
                            # 特征重要性可视化
                            if "feature_importance" in response:
                                st.subheader("特征与SOH的相关性")
                                st.image(f"data:image/png;base64,{response['feature_importance']}")
                            
                            st.session_state.features_file_id = features_file_id
                        else:
                            st.error("特征提取失败")
                else:
                    with st.spinner("正在提取特征..."):
                        # 创建特征提取器
                        extractor = BatteryFeatureExtractor(st.session_state.data)
                        
                        # 提取特征
                        if extract_time_domain:
                            extractor.extract_time_domain_features(
                                cycle_col=cycle_col,
                                voltage_col=voltage_col,
                                current_col=current_col,
                                time_col=time_col,
                                capacity_col=capacity_col
                            )
                        
                        if extract_frequency_domain:
                            extractor.extract_frequency_domain_features(
                                cycle_col=cycle_col,
                                voltage_col=voltage_col,
                                current_col=current_col,
                                time_col=time_col
                            )
                        
                        if extract_wavelet:
                            extractor.extract_wavelet_features(
                                cycle_col=cycle_col,
                                voltage_col=voltage_col,
                                current_col=current_col,
                                time_col=time_col
                            )
                        
                        if extract_ic_curve and capacity_col:
                            extractor.extract_ic_curve_features(
                                cycle_col=cycle_col,
                                voltage_col=voltage_col,
                                current_col=current_col,
                                capacity_col=capacity_col
                            )
                        
                        if extract_incremental:
                            features_df = extractor.extract_incremental_features(cycle_col)
                        else:
                            features_df = extractor.features
                        
                        # 更新会话状态
                        st.session_state.features = features_df
                        
                        # 显示提取的特征
                        st.success("特征提取完成！")
                        st.subheader("提取的特征")
                        st.dataframe(features_df.head())
                        
                        # 显示特征统计信息
                        st.subheader("特征统计信息")
                        st.info(f"共提取了 {features_df.shape[1]-1} 个特征，覆盖 {features_df.shape[0]} 个循环")
                        
                        # 特征重要性可视化
                        if 'SOH' in features_df.columns:
                            st.subheader("特征与SOH的相关性")
                            
                            # 计算与SOH的相关性
                            corr_with_soh = features_df.corr()['SOH'].sort_values(ascending=False)
                            corr_with_soh = corr_with_soh.drop('SOH')
                            
                            # 显示前10个最相关的特征
                            fig, ax = plt.subplots(figsize=(10, 6))
                            corr_with_soh.head(10).plot(kind='bar', ax=ax)
                            plt.title('与SOH最相关的10个特征')
                            plt.ylabel('相关系数')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # 保存提取的特征
                        features_file = os.path.join(OUTPUT_FOLDER, "extracted_features.csv")
                        features_df.to_csv(features_file, index=False)
                        
                        # 提供下载链接
                        with open(features_file, "rb") as file:
                            st.download_button(
                                label="下载提取的特征",
                                data=file,
                                file_name="extracted_features.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"提取特征时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回探索性分析"):
                st.session_state.current_step = 3
                st.rerun()
        with col2:
            if st.button("继续到模型训练"):
                st.session_state.current_step = 5
                st.rerun()

# 5. 模型训练页面
elif st.session_state.current_step == 5:
    st.title("5. 模型训练")
    
    if st.session_state.server_mode and not hasattr(st.session_state, 'features_file_id'):
        st.warning("请先提取特征")
        if st.button("返回特征提取"):
            st.session_state.current_step = 4
            st.rerun()
    elif not st.session_state.server_mode and st.session_state.features is None:
        st.warning("请先提取特征")
        if st.button("返回特征提取"):
            st.session_state.current_step = 4
            st.rerun()
    else:
        st.write("选择模型训练选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("目标与特征")
            
            # 选择目标列
            target_options = ["SOH"]
            if not st.session_state.server_mode and "capacity_max" in st.session_state.features.columns:
                target_options.append("capacity_max")
            
            target_col = st.selectbox("目标列", target_options)
            
            # 选择特征列
            if not st.session_state.server_mode:
                feature_cols = st.multiselect(
                    "特征列 (可选，默认使用所有数值特征)",
                    [col for col in st.session_state.features.columns if col != target_col and col != 'cycle'],
                    default=[]
                )
                
                # 如果没有选择特征，使用所有数值特征
                if not feature_cols:
                    feature_cols = [col for col in st.session_state.features.columns 
                                   if col != target_col and col != 'cycle' 
                                   and np.issubdtype(st.session_state.features[col].dtype, np.number)]
            else:
                feature_cols = st.text_input(
                    "特征列 (逗号分隔)", 
                    value="",
                    help="请输入要使用的特征列名称，以逗号分隔"
                ).split(",")
                feature_cols = [col.strip() for col in feature_cols if col.strip()]
            
            # 训练集比例
            train_ratio = st.slider("训练集比例", 0.5, 0.9, 0.8, 0.05)
        
        with col2:
            st.subheader("模型选择")
            
            model_type = st.selectbox(
                "模型类型",
                ["SVR", "随机森林", "XGBoost", "LightGBM", "LSTM"]
            )
            
            # 根据模型类型显示不同的参数
            if model_type == "SVR":
                kernel = st.selectbox("核函数", ["rbf", "linear", "poly", "sigmoid"])
                C = st.slider("正则化参数 C", 0.1, 10.0, 1.0, 0.1)
                epsilon = st.slider("Epsilon", 0.01, 0.5, 0.1, 0.01)
                model_params = {"kernel": kernel, "C": C, "epsilon": epsilon}
            
            elif model_type == "随机森林":
                n_estimators = st.slider("树的数量", 10, 200, 100, 10)
                max_depth = st.slider("最大深度", 3, 20, 10, 1)
                model_params = {"n_estimators": n_estimators, "max_depth": max_depth}
            
            elif model_type == "XGBoost":
                n_estimators = st.slider("树的数量", 10, 200, 100, 10)
                learning_rate = st.slider("学习率", 0.01, 0.3, 0.1, 0.01)
                max_depth = st.slider("最大深度", 3, 10, 6, 1)
                model_params = {
                    "n_estimators": n_estimators, 
                    "learning_rate": learning_rate,
                    "max_depth": max_depth
                }
            
            elif model_type == "LightGBM":
                n_estimators = st.slider("树的数量", 10, 200, 100, 10)
                learning_rate = st.slider("学习率", 0.01, 0.3, 0.1, 0.01)
                max_depth = st.slider("最大深度", 3, 10, 6, 1)
                model_params = {
                    "n_estimators": n_estimators, 
                    "learning_rate": learning_rate,
                    "max_depth": max_depth
                }
            
            elif model_type == "LSTM":
                units = st.slider("LSTM单元数", 16, 128, 64, 8)
                epochs = st.slider("训练轮数", 10, 200, 50, 10)
                batch_size = st.slider("批量大小", 8, 64, 32, 8)
                model_params = {
                    "units": units, 
                    "epochs": epochs,
                    "batch_size": batch_size
                }
        
        if st.button("训练模型"):
            try:
                if st.session_state.server_mode:
                    with st.spinner("正在服务器上训练模型..."):
                        request_data = {
                            "file_id": st.session_state.features_file_id,
                            "target_col": target_col,
                            "feature_cols": feature_cols,
                            "model_type": model_type,
                            "model_params": model_params,
                            "train_ratio": train_ratio
                        }
                        
                        response = send_request_to_server(
                            st.session_state.server_url,
                            "/train_model",
                            request_data,
                            st.session_state.api_key
                        )
                        
                        if response and response.get("status") == "success":
                            st.success(f"{model_type} 模型训练完成！")
                            
                            # 显示模型评估指标
                            st.subheader("模型评估")
                            metrics = response.get("metrics", {})
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("R²", f"{metrics.get('r2', 0):.4f}")
                            col2.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                            col3.metric("MSE", f"{metrics.get('mse', 0):.4f}")
                            col4.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                            
                            # 显示预测vs实际值图
                            st.subheader("预测 vs 实际值")
                            if "prediction_plot" in response:
                                st.image(f"data:image/png;base64,{response['prediction_plot']}")
                            
                            # 保存模型信息
                            st.session_state.model_id = response.get("model_id")
                            st.session_state.model_name = model_type
                            st.session_state.feature_cols = feature_cols
                            st.session_state.target_col = target_col
                        else:
                            st.error("模型训练失败")
                else:
                    with st.spinner("正在训练模型..."):
                        # 创建模型
                        model = BatteryPredictionModel()
                        
                        # 训练模型
                        model.train_model(
                            data=st.session_state.features,
                            target_col=target_col,
                            feature_cols=feature_cols,
                            model_type=model_type,
                            model_params=model_params,
                            train_ratio=train_ratio
                        )
                        
                        # 更新会话状态
                        st.session_state.model = model
                        st.session_state.model_name = model_type
                        st.session_state.feature_cols = feature_cols
                        st.session_state.target_col = target_col
                        
                        # 显示训练结果
                        st.success(f"{model_type} 模型训练完成！")
                        
                        # 显示模型评估指标
                        st.subheader("模型评估")
                        metrics = model.evaluate_model()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("R²", f"{metrics['r2']:.4f}")
                        col2.metric("MAE", f"{metrics['mae']:.4f}")
                        col3.metric("MSE", f"{metrics['mse']:.4f}")
                        col4.metric("RMSE", f"{metrics['rmse']:.4f}")
                        
                        # 显示预测vs实际值图
                        st.subheader("预测 vs 实际值")
                        fig = model.plot_prediction_vs_actual()
                        st.pyplot(fig)
                        
                        # 保存模型
                        model_file = os.path.join(MODELS_FOLDER, f"{model_type.lower()}_model.pkl")
                        joblib.dump(model, model_file)
                        
                        st.info(f"模型已保存到 {model_file}")
            
            except Exception as e:
                st.error(f"训练模型时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回特征提取"):
                st.session_state.current_step = 4
                st.rerun()
        with col2:
            if st.button("继续到预测与评估"):
                st.session_state.current_step = 6
                st.rerun()

# 6. 预测与评估页面
elif st.session_state.current_step == 6:
    st.title("6. 预测与评估")
    
    if st.session_state.server_mode and not hasattr(st.session_state, 'model_id'):
        st.warning("请先训练模型")
        if st.button("返回模型训练"):
            st.session_state.current_step = 5
            st.rerun()
    elif not st.session_state.server_mode and st.session_state.model is None:
        st.warning("请先训练模型")
        if st.button("返回模型训练"):
            st.session_state.current_step = 5
            st.rerun()
    else:
        st.write(f"使用 {st.session_state.model_name} 模型进行预测")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("SOH预测")
            
            # 选择要预测的循环
            if not st.session_state.server_mode:
                max_cycle = st.session_state.features['cycle'].max()
            else:
                max_cycle = st.number_input("当前最大循环数", min_value=1, value=100, step=1)
            
            cycles_to_predict = st.slider(
                "预测循环数",
                10,
                500,
                100,
                step=10
            )
            
            # EOL阈值
            eol_threshold = st.slider(
                "EOL阈值 (SOH百分比)",
                50, 90, 80, 1
            ) / 100.0
        
        with col2:
            st.subheader("预测选项")
            
            # 预测方法
            prediction_method = st.selectbox(
                "预测方法",
                ["直接预测", "递归预测", "集成预测"]
            )
            
            # 置信区间
            show_confidence = st.checkbox("显示置信区间", value=True)
            confidence_level = st.slider("置信水平", 0.8, 0.99, 0.95, 0.01)
        
        if st.button("执行预测"):
            try:
                if st.session_state.server_mode:
                    with st.spinner("正在服务器上进行预测..."):
                        request_data = {
                            "model_id": st.session_state.model_id,
                            "max_cycle": max_cycle,
                            "cycles_to_predict": cycles_to_predict,
                            "prediction_method": prediction_method,
                            "show_confidence": show_confidence,
                            "confidence_level": confidence_level,
                            "eol_threshold": eol_threshold
                        }
                        
                        response = send_request_to_server(
                            st.session_state.server_url,
                            "/predict",
                            request_data,
                            st.session_state.api_key
                        )
                        
                        if response and response.get("status") == "success":
                            st.subheader("SOH预测结果")
                            
                            # 显示预测图
                            if "prediction_plot" in response:
                                st.image(f"data:image/png;base64,{response['prediction_plot']}")
                            
                            # 计算RUL
                            rul = response.get("rul", 0)
                            
                            # 显示RUL
                            st.subheader("剩余使用寿命 (RUL) 预测")
                            st.info(f"预测RUL: {rul} 循环")
                            
                            # 显示RUL图
                            if "rul_plot" in response:
                                st.image(f"data:image/png;base64,{response['rul_plot']}")
                            
                            # 提供下载链接
                            predictions_file_id = response.get("file_id")
                            if predictions_file_id:
                                download_url = urljoin(
                                    st.session_state.server_url, 
                                    f"/download/{predictions_file_id}"
                                )
                                st.markdown(f"[下载预测结果]({download_url})")
                        else:
                            st.error("预测失败")
                else:
                    with st.spinner("正在预测..."):
                        model = st.session_state.model
                        
                        # 预测SOH
                        st.subheader("SOH预测结果")
                        
                        # 获取预测结果
                        predictions, confidence = model.predict_future(
                            cycles_to_predict=cycles_to_predict,
                            prediction_method=prediction_method,
                            confidence_level=confidence_level if show_confidence else None
                        )
                        
                        # 显示预测图
                        fig = model.plot_predictions(
                            predictions=predictions,
                            confidence=confidence if show_confidence else None,
                            eol_threshold=eol_threshold
                        )
                        st.pyplot(fig)
                        
                        # 计算RUL
                        rul = model.calculate_rul(
                            predictions=predictions,
                            eol_threshold=eol_threshold
                        )
                        
                        # 显示RUL
                        st.subheader("剩余使用寿命 (RUL) 预测")
                        st.info(f"预测RUL: {rul} 循环")
                        
                        # 显示RUL图
                        fig = model.plot_rul(
                            predictions=predictions,
                            eol_threshold=eol_threshold
                        )
                        st.pyplot(fig)
                        
                        # 保存预测结果
                        predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
                        pd.DataFrame({
                            'cycle': range(max_cycle + 1, max_cycle + cycles_to_predict + 1),
                            'predicted_soh': predictions
                        }).to_csv(predictions_file, index=False)
                        
                        # 提供下载链接
                        with open(predictions_file, "rb") as file:
                            st.download_button(
                                label="下载预测结果",
                                data=file,
                                file_name="predictions.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"预测时出错: {str(e)}")
        
        # 导航按钮
        col1, col2 = st.columns(2)
        with col1:
            if st.button("返回模型训练"):
                st.session_state.current_step = 5
                st.rerun()
        with col2:
            if st.button("继续到模型优化"):
                st.session_state.current_step = 7
                st.rerun()

# 7. 模型优化页面
elif st.session_state.current_step == 7:
    st.title("7. 模型优化")
    
    if st.session_state.server_mode and not hasattr(st.session_state, 'model_id'):
        st.warning("请先训练模型")
        if st.button("返回预测与评估"):
            st.session_state.current_step = 6
            st.rerun()
    elif not st.session_state.server_mode and st.session_state.model is None:
        st.warning("请先训练模型")
        if st.button("返回预测与评估"):
            st.session_state.current_step = 6
            st.rerun()
    else:
        st.write("选择模型优化选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("优化方法")
            
            optimization_method = st.selectbox(
                "优化方法",
                ["超参数优化", "特征选择", "集成学习"]
            )
            
            if optimization_method == "超参数优化":
                search_method = st.selectbox(
                    "搜索方法",
                    ["网格搜索", "随机搜索", "贝叶斯优化"]
                )
                n_iter = st.slider("搜索迭代次数", 10, 100, 30, 5)
                cv_folds = st.slider("交叉验证折数", 3, 10, 5, 1)
                
                optimization_params = {
                    "search_method": search_method,
                    "n_iter": n_iter,
                    "cv": cv_folds
                }
            
            elif optimization_method == "特征选择":
                selection_method = st.selectbox(
                    "选择方法",
                    ["递归特征消除", "特征重要性", "相关性筛选"]
                )
                n_features = st.slider(
                    "选择特征数量", 
                    5, 
                    len(st.session_state.feature_cols) if st.session_state.feature_cols else 20, 
                    min(10, len(st.session_state.feature_cols) if st.session_state.feature_cols else 10), 
                    1
                )
                
                optimization_params = {
                    "selection_method": selection_method,
                    "n_features": n_features
                }
            
            elif optimization_method == "集成学习":
                ensemble_method = st.selectbox(
                    "集成方法",
                    ["投票", "堆叠", "加权平均"]
                )
                base_models = st.multiselect(
                    "基础模型",
                    ["SVR", "随机森林", "XGBoost", "LightGBM"],
                    default=["SVR", "随机森林", "XGBoost"]
                )
                
                optimization_params = {
                    "ensemble_method": ensemble_method,
                    "base_models": base_models
                }
        
        with col2:
            st.subheader("评估选项")
            
            # 评估指标
            eval_metric = st.selectbox(
                "优化目标指标",
                ["R²", "MAE", "MSE", "RMSE"]
            )
            
            # 交叉验证
            use_cv = st.checkbox("使用交叉验证", value=True)
            
            # 可视化
            show_learning_curve = st.checkbox("显示学习曲线", value=True)
        
        if st.button("执行模型优化"):
            try:
                if st.session_state.server_mode:
                    with st.spinner("正在服务器上优化模型..."):
                        request_data = {
                            "model_id": st.session_state.model_id,
                            "optimization_method": optimization_method,
                            "optimization_params": optimization_params,
                            "eval_metric": eval_metric,
                            "use_cv": use_cv
                        }
                        
                        response = send_request_to_server(
                            st.session_state.server_url,
                            "/optimize_model",
                            request_data,
                            st.session_state.api_key
                        )
                        
                        if response and response.get("status") == "success":
                            st.success("模型优化完成！")
                            
                            # 显示优化后的评估指标
                            st.subheader("优化后的模型评估")
                            metrics = response.get("metrics", {})
                            
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("R²", f"{metrics.get('r2', 0):.4f}")
                            col2.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                            col3.metric("MSE", f"{metrics.get('mse', 0):.4f}")
                            col4.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                            
                            # 显示预测vs实际值图
                            st.subheader("预测 vs 实际值")
                            if "prediction_plot" in response:
                                st.image(f"data:image/png;base64,{response['prediction_plot']}")
                            
                            # 显示学习曲线
                            if show_learning_curve and "learning_curve" in response:
                                st.subheader("学习曲线")
                                st.image(f"data:image/png;base64,{response['learning_curve']}")
                            
                            # 更新模型ID
                            st.session_state.model_id = response.get("optimized_model_id", st.session_state.model_id)
                        else:
                            st.error("模型优化失败")
                else:
                    with st.spinner("正在优化模型..."):
                        model = st.session_state.model
                        evaluator = ModelEvaluator(model)
                        
                        # 执行优化
                        if optimization_method == "超参数优化":
                            optimized_model = evaluator.optimize_hyperparameters(
                                search_method=optimization_params["search_method"],
                                n_iter=optimization_params["n_iter"],
                                cv=optimization_params["cv"],
                                scoring=eval_metric.lower()
                            )
                        
                        elif optimization_method == "特征选择":
                            optimized_model = evaluator.select_features(
                                method=optimization_params["selection_method"],
                                n_features=optimization_params["n_features"]
                            )
                        
                        elif optimization_method == "集成学习":
                            optimized_model = evaluator.build_ensemble(
                                method=optimization_params["ensemble_method"],
                                base_models=optimization_params["base_models"]
                            )
                        
                        # 更新会话状态
                        st.session_state.model = optimized_model
                        
                        # 显示优化结果
                        st.success("模型优化完成！")
                        
                        # 显示优化后的评估指标
                        st.subheader("优化后的模型评估")
                        metrics = optimized_model.evaluate_model()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("R²", f"{metrics['r2']:.4f}")
                        col2.metric("MAE", f"{metrics['mae']:.4f}")
                        col3.metric("MSE", f"{metrics['mse']:.4f}")
                        col4.metric("RMSE", f"{metrics['rmse']:.4f}")
                        
                        # 显示预测vs实际值图
                        st.subheader("预测 vs 实际值")
                        fig = optimized_model.plot_prediction_vs_actual()
                        st.pyplot(fig)
                        
                        # 显示学习曲线
                        if show_learning_curve:
                            st.subheader("学习曲线")
                            fig = evaluator.plot_learning_curve(cv=5 if use_cv else None)
                            st.pyplot(fig)
                        
                        # 保存优化后的模型
                        model_file = os.path.join(MODELS_FOLDER, "optimized_model.pkl")
                        joblib.dump(optimized_model, model_file)
                        
                        st.info(f"优化后的模型已保存到 {model_file}")
            
            except Exception as e:
                st.error(f"优化模型时出错: {str(e)}")
        
        # 导航按钮
        if st.button("返回预测与评估"):
            st.session_state.current_step = 6
            st.rerun()

# 页脚
st.markdown("---")
st.markdown("### 电池寿命预测系统 | 基于机器学习的SOH和RUL预测")
st.markdown("© 2025 浙江锋锂新能源科技有限公司-唐光盛团队")

if st.session_state.server_mode:
    st.markdown(f"**当前模式:** 服务器模式 | **服务器地址:** {st.session_state.server_url}")
else:
    st.markdown("**当前模式:** 本地模式")
