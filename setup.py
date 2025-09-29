"""
setup.py - Project Installation Configuration

Define project metadata and installation configuration.
"""

from setuptools import setup, find_packages
import os

def read_requirements():
    """Read requirements.txt file"""
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if not os.path.exists(requirements_path):
        return []
    
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f.readlines() 
            if line.strip() and not line.startswith('#')
        ]
    
    # Filter out actual package dependencies (exclude empty lines and comments)
    package_requirements = []
    for req in requirements:
        # Skip comment lines starting with #
        if req.startswith('#'):
            continue
        # Skip empty lines
        if not req.strip():
            continue
        # Skip optional dependencies (lines starting with a space and then #)
        if req.startswith(' ') and req.lstrip().startswith('#'):
            continue
        package_requirements.append(req)
    
    return package_requirements

def read_readme():
    """Read README.md file"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return "共享单车使用模式聚类分析项目"

# Read version information (if version.txt exists)
def get_version():
    version_path = os.path.join(os.path.dirname(__file__), 'VERSION')
    if os.path.exists(version_path):
        with open(version_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return '2.0.0'

# Project configuration
setup(
    name='shared_bikes_analysis',
    version=get_version(),
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for shared bikes usage pattern analysis and clustering',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/shared_bikes',
    packages=find_packages(exclude=['tests*', 'docs*', '.venv*', 'notebooks*']),
    package_dir={
        'shared_bikes': '.',
    },
    package_data={
        'shared_bikes': [
            'configs/*.yaml',
            'configs/*.py',
            'data/*',
            'docs/*',
            'models/*',
            'logs/*',
            'output/*',
        ],
    },
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
            'jupyter>=1.0.0'
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=0.5.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'shared-bikes-run=scripts.run_pipeline:main',
            'shared-bikes-train=scripts.train_model:main',
            'shared-bikes-predict=scripts.predict:main',
            'shared-bikes-evaluate=scripts.evaluate_model:main',
            'shared-bikes-setup=scripts.setup_data:main'
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.8',
    keywords='shared bikes, clustering, machine learning, data analysis',
)