from setuptools import find_packages, setup

setup(
    name="circumnutation_pipeline",
    packages=find_packages(exclude=["circumnutation_pipeline_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
