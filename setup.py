from setuptools import setup

setup(
    name="jtorch",
    version="0.1",
    packages=["jtorch"],
    package_data={"jtorch": []},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
