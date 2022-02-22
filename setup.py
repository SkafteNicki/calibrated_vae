#!/usr/bin/env python
import os
from io import open

from setuptools import Command, find_packages, setup


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")


with open("requirements.txt", "r") as reqs:
    requirements = reqs.read().split()


setup(
    name='mixer_ensemble',
    version="0.1",
    author="Nicki Skafte Detlefsen",
    author_email="nsde@dtu.dk",
    packages=find_packages(exclude=["tests", "tests/*"]),
    python_requires=">=3.8",
    install_requires=requirements,
    cmdclass={"clean": CleanCommand},
)