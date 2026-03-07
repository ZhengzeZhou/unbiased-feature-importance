from setuptools import setup, find_packages

setup(
    name='unbiased-feature-importance',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['numpy', 'scikit-learn'],
    python_requires='>=3.7',
    description='Unbiased feature importance for Random Forests (arXiv:1903.05179)',
    author='Zhengze Zhou',
    license='MIT',
)
