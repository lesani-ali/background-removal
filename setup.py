from setuptools import setup, find_packages


with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name='background_removal',
    version='0.1',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    include_package_data=True,
    description='A tool for removing background'
)
