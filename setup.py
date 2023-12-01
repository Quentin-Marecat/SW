#!/usr/bin/env python3
from setuptools import setup
from setuptools.command.install import install
import subprocess

class InstallCommand(install):
    def run(self):
        # Exécute la méthode d'installation de la classe parente
        install.run(self)

        # Installez vos dépendances ici si elles ne sont pas déjà présentes
        dependencies = [
        "numpy>=1.19.0",
        "scipy>=1.1.0",
        "qiskit>=0.45.0",
        "qiskit_nature==0.4.4",
        "qiskit_algorithms>=0.2.1",
        "qiskit_aer>=0.13.0"
        ]
        for dependency in dependencies:
            try:
                __import__(dependency)
            except ImportError:
                subprocess.call(['pip', 'install', dependency])

# Configuration du package
setup(
    name='SW',
    version='1.0.0',
    author='Quentin Marecat',
    author_email='quentin.marecat@gmail.com',
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.1.0",
        "qiskit>=0.45.0",
        "qiskit_nature==0.4.4",
        "qiskit_algorithms>=0.2.1",
        "qiskit_aer>=0.13.0"
    ],
    cmdclass={'install': InstallCommand},
)
