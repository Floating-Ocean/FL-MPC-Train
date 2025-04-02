from setuptools import setup, find_packages
setup(
    name="fl_mpc_training",
    version="0.1",
    packages=find_packages(),
    install_requires=[],
    extras_require={
        'dev': ['pytest']
    }
)
