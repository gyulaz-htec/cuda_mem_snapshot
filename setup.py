from setuptools import setup, find_packages

setup(
    name='cuda_mem_snapshot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.4.0'  # Specify the required PyTorch version
    ],
    description='A small library providing CUDA memory management utilities.',
    author='Gyula Zakor',
    author_email='gyula.zakor@htecgroup.com',
    url='https://github.com/gyulaz-htec/cuda_mem_snapshot',  # Optional
)
