"""
服务器连接管理器
支持SSH连接、文件传输和GitHub集成
"""

import paramiko
import os
import io
import json
import time
import threading
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import git
import tempfile
import shutil
import streamlit as st
from cryptography.fernet import Fernet
import base64
import hashlib

class ServerConnectionManager:
    """服务器连接管理器"""
    
    def __init__(self):
        self.ssh_client = None
        self.sftp_client = None
        self.connection_status = "disconnected"
        self.last_error = None
        self.connection_config = {}
        self.github_config = {}
        self._encryption_key = None
        
    def _get_encryption_key(self) -> bytes:
        """获取或生成加密密钥"""
        if self._encryption_key is None:
            # 使用会话状态存储密钥，确保在会话期间保持一致
            if 'encryption_key' not in st.session_state:
                st.session_state.encryption_key = Fernet.generate_key()
            self._encryption_key = st.session_state.encryption_key
        return self._encryption_key
    
    def _encrypt_data(self, data: str) -> str:
        """加密敏感数据"""
        try:
            f = Fernet(self._get_encryption_key())
            encrypted_data = f.encrypt(data.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            st.error(f"数据加密失败: {e}")
            return data
    
    def _decrypt_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        try:
            f = Fernet(self._get_encryption_key())
            decoded_data = base64.b64decode(encrypted_data.encode())
            decrypted_data = f.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            st.error(f"数据解密失败: {e}")
            return encrypted_data
    
    def save_connection_config(self, config: Dict[str, Any]) -> bool:
        """保存连接配置（加密敏感信息）"""
        try:
            # 加密敏感信息
            encrypted_config = config.copy()
            if 'password' in encrypted_config and encrypted_config['password']:
                encrypted_config['password'] = self._encrypt_data(encrypted_config['password'])
            if 'private_key_content' in encrypted_config and encrypted_config['private_key_content']:
                encrypted_config['private_key_content'] = self._encrypt_data(encrypted_config['private_key_content'])
            
            # 保存到会话状态
            st.session_state.server_config = encrypted_config
            self.connection_config = config  # 保存未加密版本用于当前会话
            return True
        except Exception as e:
            self.last_error = f"保存配置失败: {e}"
            return False
    
    def load_connection_config(self) -> Dict[str, Any]:
        """加载连接配置（解密敏感信息）"""
        try:
            if 'server_config' in st.session_state:
                encrypted_config = st.session_state.server_config.copy()
                
                # 解密敏感信息
                if 'password' in encrypted_config and encrypted_config['password']:
                    encrypted_config['password'] = self._decrypt_data(encrypted_config['password'])
                if 'private_key_content' in encrypted_config and encrypted_config['private_key_content']:
                    encrypted_config['private_key_content'] = self._decrypt_data(encrypted_config['private_key_content'])
                
                self.connection_config = encrypted_config
                return encrypted_config
            return {}
        except Exception as e:
            self.last_error = f"加载配置失败: {e}"
            return {}
    
    def test_ssh_connection(self, host: str, port: int, username: str, 
                           password: str = None, private_key_path: str = None,
                           private_key_content: str = None) -> Tuple[bool, str]:
        """测试SSH连接"""
        try:
            test_client = paramiko.SSHClient()
            test_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 准备认证参数
            auth_kwargs = {
                'hostname': host,
                'port': port,
                'username': username,
                'timeout': 10
            }
            
            # 选择认证方式
            if private_key_content:
                # 使用私钥内容
                key_file = io.StringIO(private_key_content)
                try:
                    private_key = paramiko.RSAKey.from_private_key(key_file)
                    auth_kwargs['pkey'] = private_key
                except:
                    try:
                        key_file.seek(0)
                        private_key = paramiko.Ed25519Key.from_private_key(key_file)
                        auth_kwargs['pkey'] = private_key
                    except:
                        key_file.seek(0)
                        private_key = paramiko.ECDSAKey.from_private_key(key_file)
                        auth_kwargs['pkey'] = private_key
            elif private_key_path and os.path.exists(private_key_path):
                # 使用私钥文件
                try:
                    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)
                    auth_kwargs['pkey'] = private_key
                except:
                    try:
                        private_key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
                        auth_kwargs['pkey'] = private_key
                    except:
                        private_key = paramiko.ECDSAKey.from_private_key_file(private_key_path)
                        auth_kwargs['pkey'] = private_key
            elif password:
                # 使用密码认证
                auth_kwargs['password'] = password
            else:
                return False, "未提供有效的认证信息"
            
            # 尝试连接
            test_client.connect(**auth_kwargs)
            
            # 测试执行命令
            stdin, stdout, stderr = test_client.exec_command('echo "连接测试成功"')
            result = stdout.read().decode().strip()
            
            test_client.close()
            
            if result == "连接测试成功":
                return True, "SSH连接测试成功"
            else:
                return False, "命令执行失败"
                
        except paramiko.AuthenticationException:
            return False, "认证失败，请检查用户名、密码或私钥"
        except paramiko.SSHException as e:
            return False, f"SSH连接错误: {e}"
        except Exception as e:
            return False, f"连接失败: {e}"
    
    def connect_ssh(self, host: str, port: int, username: str,
                   password: str = None, private_key_path: str = None,
                   private_key_content: str = None) -> bool:
        """建立SSH连接"""
        try:
            # 关闭现有连接
            self.disconnect()
            
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            # 准备认证参数
            auth_kwargs = {
                'hostname': host,
                'port': port,
                'username': username,
                'timeout': 30
            }
            
            # 选择认证方式
            if private_key_content:
                # 使用私钥内容
                key_file = io.StringIO(private_key_content)
                try:
                    private_key = paramiko.RSAKey.from_private_key(key_file)
                    auth_kwargs['pkey'] = private_key
                except:
                    try:
                        key_file.seek(0)
                        private_key = paramiko.Ed25519Key.from_private_key(key_file)
                        auth_kwargs['pkey'] = private_key
                    except:
                        key_file.seek(0)
                        private_key = paramiko.ECDSAKey.from_private_key(key_file)
                        auth_kwargs['pkey'] = private_key
            elif private_key_path and os.path.exists(private_key_path):
                # 使用私钥文件
                try:
                    private_key = paramiko.RSAKey.from_private_key_file(private_key_path)
                    auth_kwargs['pkey'] = private_key
                except:
                    try:
                        private_key = paramiko.Ed25519Key.from_private_key_file(private_key_path)
                        auth_kwargs['pkey'] = private_key
                    except:
                        private_key = paramiko.ECDSAKey.from_private_key_file(private_key_path)
                        auth_kwargs['pkey'] = private_key
            elif password:
                # 使用密码认证
                auth_kwargs['password'] = password
            else:
                self.last_error = "未提供有效的认证信息"
                return False
            
            # 建立连接
            self.ssh_client.connect(**auth_kwargs)
            
            # 建立SFTP连接
            self.sftp_client = self.ssh_client.open_sftp()
            
            self.connection_status = "connected"
            self.last_error = None
            
            # 保存连接配置
            self.connection_config = {
                'host': host,
                'port': port,
                'username': username,
                'password': password,
                'private_key_path': private_key_path,
                'private_key_content': private_key_content
            }
            
            return True
            
        except Exception as e:
            self.last_error = f"连接失败: {e}"
            self.connection_status = "error"
            return False
    
    def disconnect(self):
        """断开连接"""
        try:
            if self.sftp_client:
                self.sftp_client.close()
                self.sftp_client = None
            
            if self.ssh_client:
                self.ssh_client.close()
                self.ssh_client = None
            
            self.connection_status = "disconnected"
            self.last_error = None
            
        except Exception as e:
            self.last_error = f"断开连接时出错: {e}"
    
    def execute_command(self, command: str) -> Tuple[bool, str, str]:
        """执行远程命令"""
        if not self.ssh_client or self.connection_status != "connected":
            return False, "", "未建立SSH连接"
        
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            
            # 读取输出
            stdout_data = stdout.read().decode('utf-8', errors='ignore')
            stderr_data = stderr.read().decode('utf-8', errors='ignore')
            
            # 获取退出状态
            exit_status = stdout.channel.recv_exit_status()
            
            return exit_status == 0, stdout_data, stderr_data
            
        except Exception as e:
            return False, "", f"命令执行失败: {e}"
    
    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """上传文件到远程服务器"""
        if not self.sftp_client or self.connection_status != "connected":
            self.last_error = "未建立SFTP连接"
            return False
        
        try:
            # 确保远程目录存在
            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self.execute_command(f"mkdir -p {remote_dir}")
            
            self.sftp_client.put(local_path, remote_path)
            return True
            
        except Exception as e:
            self.last_error = f"文件上传失败: {e}"
            return False
    
    def download_file(self, remote_path: str, local_path: str) -> bool:
        """从远程服务器下载文件"""
        if not self.sftp_client or self.connection_status != "connected":
            self.last_error = "未建立SFTP连接"
            return False
        
        try:
            # 确保本地目录存在
            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)
            
            self.sftp_client.get(remote_path, local_path)
            return True
            
        except Exception as e:
            self.last_error = f"文件下载失败: {e}"
            return False
    
    def list_remote_directory(self, remote_path: str) -> List[Dict[str, Any]]:
        """列出远程目录内容"""
        if not self.sftp_client or self.connection_status != "connected":
            return []
        
        try:
            files = []
            for item in self.sftp_client.listdir_attr(remote_path):
                files.append({
                    'name': item.filename,
                    'size': item.st_size,
                    'modified': time.ctime(item.st_mtime),
                    'is_dir': item.st_mode & 0o040000 != 0
                })
            return files
            
        except Exception as e:
            self.last_error = f"列出目录失败: {e}"
            return []
    
    def clone_github_repo(self, repo_url: str, local_path: str, 
                         branch: str = "main", token: str = None) -> bool:
        """克隆GitHub仓库"""
        try:
            # 如果目录已存在，先删除
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            
            # 准备认证URL
            if token and repo_url.startswith('https://github.com/'):
                # 添加token到URL
                auth_url = repo_url.replace('https://github.com/', f'https://{token}@github.com/')
            else:
                auth_url = repo_url
            
            # 克隆仓库
            repo = git.Repo.clone_from(auth_url, local_path, branch=branch)
            
            self.github_config = {
                'repo_url': repo_url,
                'local_path': local_path,
                'branch': branch,
                'token': token
            }
            
            return True
            
        except Exception as e:
            self.last_error = f"克隆仓库失败: {e}"
            return False
    
    def update_github_repo(self, local_path: str = None) -> bool:
        """更新GitHub仓库"""
        try:
            if not local_path:
                local_path = self.github_config.get('local_path')
            
            if not local_path or not os.path.exists(local_path):
                self.last_error = "仓库路径不存在"
                return False
            
            repo = git.Repo(local_path)
            origin = repo.remotes.origin
            origin.pull()
            
            return True
            
        except Exception as e:
            self.last_error = f"更新仓库失败: {e}"
            return False
    
    def sync_github_to_server(self, local_repo_path: str, remote_server_path: str) -> bool:
        """将GitHub仓库同步到远程服务器"""
        if not self.sftp_client or self.connection_status != "connected":
            self.last_error = "未建立服务器连接"
            return False
        
        try:
            # 确保远程目录存在
            self.execute_command(f"mkdir -p {remote_server_path}")
            
            # 递归上传文件
            def upload_directory(local_dir, remote_dir):
                for item in os.listdir(local_dir):
                    if item.startswith('.git'):
                        continue  # 跳过git文件
                    
                    local_item_path = os.path.join(local_dir, item)
                    remote_item_path = f"{remote_dir}/{item}"
                    
                    if os.path.isdir(local_item_path):
                        # 创建远程目录
                        self.execute_command(f"mkdir -p {remote_item_path}")
                        # 递归上传子目录
                        upload_directory(local_item_path, remote_item_path)
                    else:
                        # 上传文件
                        self.sftp_client.put(local_item_path, remote_item_path)
            
            upload_directory(local_repo_path, remote_server_path)
            return True
            
        except Exception as e:
            self.last_error = f"同步到服务器失败: {e}"
            return False
    
    def get_connection_status(self) -> Dict[str, Any]:
        """获取连接状态信息"""
        return {
            'status': self.connection_status,
            'last_error': self.last_error,
            'config': self.connection_config,
            'github_config': self.github_config
        }
    
    def find_data_files(self, directory: str, extensions: List[str] = None) -> List[str]:
        """在目录中查找数据文件"""
        if extensions is None:
            extensions = ['.xlsx', '.xls', '.csv', '.json']
        
        data_files = []
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        data_files.append(os.path.join(root, file))
            
            return data_files
            
        except Exception as e:
            self.last_error = f"查找数据文件失败: {e}"
            return []
    
    def get_remote_data_files(self, remote_directory: str, 
                            extensions: List[str] = None) -> List[str]:
        """获取远程服务器上的数据文件列表"""
        if not self.ssh_client or self.connection_status != "connected":
            return []
        
        if extensions is None:
            extensions = ['.xlsx', '.xls', '.csv', '.json']
        
        try:
            # 构建find命令
            ext_pattern = " -o ".join([f'-name "*.{ext.lstrip(".")}"' for ext in extensions])
            command = f'find {remote_directory} -type f \\( {ext_pattern} \\)'
            
            success, stdout, stderr = self.execute_command(command)
            
            if success:
                files = [line.strip() for line in stdout.split('\n') if line.strip()]
                return files
            else:
                self.last_error = f"查找远程文件失败: {stderr}"
                return []
                
        except Exception as e:
            self.last_error = f"获取远程文件列表失败: {e}"
            return []


class GitHubDataManager:
    """GitHub数据管理器"""
    
    def __init__(self, connection_manager: ServerConnectionManager):
        self.connection_manager = connection_manager
        self.temp_dir = None
    
    def setup_temp_directory(self) -> str:
        """设置临时目录"""
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="battery_data_")
        return self.temp_dir
    
    def cleanup_temp_directory(self):
        """清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def fetch_data_from_github(self, repo_url: str, branch: str = "main", 
                              token: str = None) -> Tuple[bool, List[str]]:
        """从GitHub获取数据文件"""
        try:
            # 设置临时目录
            temp_dir = self.setup_temp_directory()
            repo_dir = os.path.join(temp_dir, "repo")
            
            # 克隆仓库
            success = self.connection_manager.clone_github_repo(
                repo_url, repo_dir, branch, token
            )
            
            if not success:
                return False, []
            
            # 查找数据文件
            data_files = self.connection_manager.find_data_files(repo_dir)
            
            return True, data_files
            
        except Exception as e:
            self.connection_manager.last_error = f"从GitHub获取数据失败: {e}"
            return False, []
    
    def sync_to_server(self, server_path: str) -> bool:
        """将数据同步到服务器"""
        if not self.temp_dir:
            self.connection_manager.last_error = "没有可同步的数据"
            return False
        
        repo_dir = os.path.join(self.temp_dir, "repo")
        return self.connection_manager.sync_github_to_server(repo_dir, server_path)


# 全局连接管理器实例
if 'connection_manager' not in st.session_state:
    st.session_state.connection_manager = ServerConnectionManager()

if 'github_manager' not in st.session_state:
    st.session_state.github_manager = GitHubDataManager(st.session_state.connection_manager)

