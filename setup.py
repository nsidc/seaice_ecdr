from setuptools import find_packages, setup

setup(
    name='seaice_ecdr',
    version='0.1.0',
    description=('Sea ice concentration estimates for NSIDC/NOAA CDR'),
    url='https://github.com/nsidc/seaice_ecdr',
    author='NSIDC Development Team',
    license='MIT',
    packages=find_packages(
        exclude=(
            '*.tasks',
            '*.tasks.*',
            'tasks.*',
            'tasks',
        ),
    ),
    include_package_data=True,
    zip_safe=False,
)
