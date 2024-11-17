from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(name="sql-rl-gen", include_package_data=True, python_requires=">=3.7", install_requires=parse_requirements("requirements.txt"), packages=find_packages("."))