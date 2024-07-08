from setuptools import setup, find_packages

setup(
    name='MP3TOMIDI',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pretty_midi',
        'librosa',
        'mir_eval',
        'tables',
        'torch',
        'torchvision',
        'torchaudio'
    ],
)
