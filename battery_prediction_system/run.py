#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电池寿命预测模型 - Web应用启动脚本
该脚本用于启动电池寿命预测模型的Web应用。
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # 确保目录存在
    for folder in ['uploads', 'output', 'models', 'templates']:
        os.makedirs(folder, exist_ok=True)
    
    # 启动应用
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
