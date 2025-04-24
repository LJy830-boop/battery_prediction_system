# 电池寿命预测系统

这是一个基于机器学习的电池寿命预测系统，可以预测电池的健康状态(SOH)和剩余使用寿命(RUL)。

## 功能特点

- **数据预处理**：支持CSV和Excel格式的电池数据上传，自动清洗和标准化
- **探索性数据分析**：生成数据摘要、分布图、相关性矩阵和容量退化曲线
- **特征提取**：实现了多种特征提取方法，包括时域特征、频域特征、小波特征和IC曲线特征
- **模型训练**：支持多种机器学习和深度学习模型，包括SVR、随机森林、XGBoost、LightGBM和LSTM
- **预测与评估**：可预测SOH和RUL，并提供可视化结果
- **模型优化**：支持超参数优化、特征选择和集成学习

## 技术亮点

1. **多种特征提取方法**：实现了时域、频域、小波和IC曲线等多种特征提取方法，提高了预测精度
2. **先进模型支持**：集成了最新的机器学习和深度学习模型
3. **完整的评估框架**：提供了全面的模型评估和优化工具
4. **用户友好界面**：基于Streamlit的直观步骤式界面，支持数据可视化
5. **完整的部署方案**：支持本地运行和云端部署

## 安装说明

### 环境要求

- Python 3.8+
- 依赖库：见`requirements.txt`

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/battery-prediction.git
cd battery-prediction
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 运行应用
```bash
streamlit run streamlit_app.py
```

## 使用指南

1. **数据上传**：上传电池数据文件（CSV或Excel格式）
2. **数据预处理**：选择预处理选项，如移除异常值、填充缺失值等
3. **探索性分析**：查看数据分布、相关性和容量退化曲线
4. **特征提取**：选择要提取的特征类型，如时域特征、频域特征等
5. **模型训练**：选择模型类型和参数，训练预测模型
6. **预测与评估**：输入循环次数和EOL阈值，获取SOH和RUL预测结果
7. **模型优化**：选择优化方法，如超参数优化、特征选择等

## 项目结构

```
battery_prediction/
├── data_preprocessing_pipeline.py  # 数据预处理模块
├── exploratory_data_analysis.py    # 探索性数据分析模块
├── feature_extraction.py           # 特征提取模块
├── prediction_models.py            # 预测模型模块
├── model_evaluation.py             # 模型评估与优化模块
├── streamlit_app.py                # Streamlit应用主程序
├── requirements.txt                # 依赖库列表
├── models/                         # 保存训练好的模型
├── uploads/                        # 上传的数据文件
└── output/                         # 输出结果和图表
```

## 数据格式要求

上传的数据文件应包含以下列：
- 循环次数（cycle）
- 电压（voltage）
- 电流（current）
- 时间（time）
- 容量（capacity，可选）
- 温度（temperature，可选）

## 贡献指南

欢迎提交问题和拉取请求。对于重大更改，请先开issue讨论您想要更改的内容。

## 许可证

[MIT](https://choosealicense.com/licenses/mit/)
