from setuptools import find_packages,setup


HYPEN_E_DOT="-e ."
from typing import List
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as obj:
        requirements=obj.readlines()    
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements    



setup(
    name="Regressor_Project",
    version="0.0.1",
    author="Satyajit",
    author_email="satyajitsamal198076@gmail.com",
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages()

      )