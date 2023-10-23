from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path):
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        return requirements 

setup(name = 'Flight-fare-prediction',
      version= '0.0.1',
      author = 'Yash',
      author_email='yashkeshari79@gmail.com',
      install_requires = get_requirements('requirements.txt'),
      packages = find_packages()
      )
