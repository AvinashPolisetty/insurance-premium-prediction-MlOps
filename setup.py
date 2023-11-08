from setuptools import setup,find_packages
from typing import List

Hypen_e_dot="-e ."

def get_requirements(file_path:str)->List[str]:
    requirements=[]

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[ req.replace("\n","") for req in requirements]

        if Hypen_e_dot in requirements:
            requirements.remove(Hypen_e_dot)

    return requirements


setup(
    name="Insurance-Premium-Prediction",
    version="0.0.1",
    author="Avinash",
    author_email="aviavinashp64@gmail.com",
    description="Creating a application which will predict how much of premium they have to pay for insurance",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)