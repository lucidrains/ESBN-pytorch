from setuptools import setup, find_packages

setup(
  name = 'esbn-pytorch',
  packages = find_packages(),
  version = '0.0.4',
  license='MIT',
  description = 'Emergent Symbol Binding Network - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/ESBN-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'neuro-symbolic',
    'abstract reasoning',
    'memory'
  ],
  install_requires=[
    'torch>=1.6',
    'einops>=0.3'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)