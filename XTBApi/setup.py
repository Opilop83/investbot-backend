from setuptools import setup, find_packages

setup(
    name='XTBApi',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'websocket-client',
        'requests',
    ],
    author='Opilop83',
    description='XTB WebSocket API wrapper for InvestBot',
)
