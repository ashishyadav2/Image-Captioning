from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT = '-e .'
def get_requirements(file_path: str)-> List[str]:
    '''
    this function will return list of requirements
    '''
    requirements = []
    with open(file_path,'r') as file_obj:
        requirements = file_obj.read().split('\n')
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements 

setup(
    name='Image-Captioning',
    version='0.0.1',
    author='Hydrogen Cyanide',
    author_email = '--',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)