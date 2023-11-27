from setuptools import find_packages, setup
from typing import List

HYPHE_E_DOT = "-e ."

def get_requirements(file_path)->List[str]:
    '''
    this funtion will return the list
    '''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHE_E_DOT in requirements:
            requirements.remove(HYPHE_E_DOT)
    return requirements




setup(
    name = 'mlporject',
    version = '0.0.1',
    author = 'Sharath chandara b',
    author_email= 'schandrab22@gmail.com',
    packages = find_packages(),
    install_requires=get_requirements('requirements.txt')

)