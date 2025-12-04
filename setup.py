import pkg_resources
from setuptools import find_packages, setup

setup(
    name="steam_content_based_recsys",
    py_modules=["steam_content_based_recsys"],
    version="0.1.0",
    description="Content Based RecSys for coursework.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    author="Maksim-Goncharovskiy",
    url="https://github.com/Maksim-Goncharovskiy/steam-content-based-recsys",
    license="Unlicense",
    packages=find_packages(include=["steam_content_based_recsys"]),
    python_requires=">=3.10",
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open("requirements.txt", "r", encoding="utf-8").read()
        )
    ],
    include_package_data=True,
)