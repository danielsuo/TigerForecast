# sets up package

import setuptools

# 0.xx = alpha, 1.xx = beta, >2.xx = full release
version = '0.1'

'''
Cython>=0.22
pystan>=2.14
numpy>=1.10.0
pandas>=0.23.4
matplotlib>=2.0.0
LunarCalendar>=0.0.9
convertdate>=2.1.2
holidays>=0.9.5
setuptools-git>=1.2
'''

extras = {
  'prophet': ['prophet', 'Cython', 'pystan', 'LunarCalendar', 'convertdate', 'holidays', 'setuptools-git'],
}
extras['all'] = [item for group in extras.values() for item in group]

setuptools.setup(
    name='tigerforecast',
    url='https://github.com/MinRegret/TigerForecast',
    author='Google AI Princeton',
    author_email='johnolof@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=open('requirements.txt').read().split('\n'),
    extras_require=extras,
    version=version,
    license='Apache License 2.0',
    description='Princeton time-series framework',
    long_description=open('README.md').read(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: Apache License 2.0",
		"Operating System :: OS Independent",
	],
)
