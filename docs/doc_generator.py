"""
文档生成器模块
提供自动化文档生成功能
"""

import os
import ast
from pathlib import Path
from typing import List, Dict, Any


class DocGenerator:
    """文档生成器类"""
    
    def __init__(self, source_dir: str = "src", docs_dir: str = "docs"):
        """
        初始化文档生成器
        
        Args:
            source_dir (str): 源代码目录
            docs_dir (str): 文档输出目录
        """
        self.source_dir = Path(source_dir)
        self.docs_dir = Path(docs_dir)
        
    def _parse_python_file(self, file_path: Path) -> Dict[str, Any]:
        """
        解析Python文件，提取文档信息
        
        Args:
            file_path (Path): Python文件路径
            
        Returns:
            Dict: 包含文件文档信息的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                tree = ast.parse(file.read())
                
            doc_info = {
                'filename': file_path.name,
                'module_docstring': None,
                'classes': [],
                'functions': []
            }
            
            # 提取模块文档字符串
            if ast.get_docstring(tree):
                doc_info['module_docstring'] = ast.get_docstring(tree)
                
            # 遍历AST节点
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'methods': []
                    }
                    
                    # 提取类方法
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_info = {
                                'name': item.name,
                                'docstring': ast.get_docstring(item),
                                'args': [arg.arg for arg in item.args.args]
                            }
                            class_info['methods'].append(method_info)
                            
                    doc_info['classes'].append(class_info)
                    
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'docstring': ast.get_docstring(node),
                        'args': [arg.arg for arg in node.args.args]
                    }
                    doc_info['functions'].append(func_info)
                    
            return doc_info
            
        except Exception as e:
            print(f"解析文件 {file_path} 失败: {e}")
            return {}
    
    def generate_api_docs(self) -> str:
        """生成API文档内容"""
        api_docs = "# API文档\n\n"
        
        # 遍历src目录下的所有Python文件
        python_files = list(self.source_dir.rglob("*.py"))
        python_files.sort()
        
        for file_path in python_files:
            # 跳过__pycache__和测试文件
            if '__pycache__' in str(file_path) or 'test' in str(file_path).lower():
                continue
                
            relative_path = file_path.relative_to(self.source_dir)
            doc_info = self._parse_python_file(file_path)
            
            if not doc_info:
                continue
                
            # 添加文件标题
            api_docs += f"## {relative_path}\n\n"
            
            # 添加模块文档字符串
            if doc_info['module_docstring']:
                api_docs += f"{doc_info['module_docstring']}\n\n"
            
            # 添加类信息
            for class_info in doc_info['classes']:
                api_docs += f"### 类: {class_info['name']}\n\n"
                if class_info['docstring']:
                    api_docs += f"{class_info['docstring']}\n\n"
                
                # 添加方法信息
                for method in class_info['methods']:
                    args_str = ", ".join(method['args']) if method['args'] else ""
                    api_docs += f"#### 方法: {method['name']}({args_str})\n\n"
                    if method['docstring']:
                        api_docs += f"{method['docstring']}\n\n"
            
            # 添加函数信息
            for func in doc_info['functions']:
                args_str = ", ".join(func['args']) if func['args'] else ""
                api_docs += f"### 函数: {func['name']}({args_str})\n\n"
                if func['docstring']:
                    api_docs += f"{func['docstring']}\n\n"
            
        return api_docs
    
    def update_documentation(self) -> None:
        """更新项目文档"""
        # 生成API文档
        api_docs = self.generate_api_docs()
        
        # 读取现有文档
        readme_path = self.docs_dir / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
            # 分割文档内容
            parts = content.split("## API文档\n\n")
            if len(parts) > 1:
                # 保留API文档之前的内容
                updated_content = parts[0] + "## API文档\n\n" + api_docs
                
                # 写回文件
                with open(readme_path, 'w', encoding='utf-8') as file:
                    file.write(updated_content)
                
                print(f"文档已更新: {readme_path}")
            else:
                print("无法找到API文档部分，跳过更新")
        else:
            print(f"找不到文档文件: {readme_path}")