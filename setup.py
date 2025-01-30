from setuptools import setup, find_packages

setup(
    name="SK",
    version="1.0.0",
    packages=find_packages(),
    description="A Python implementation of the card game Skull King. Owned by Grandpa Beck's Games. We have no affiliation with Grandpa Beck's Games.",
    entry_points={
        "console_scripts": [
            "skullking=SK.main:main",
        ],
    },
)