# -*- coding: utf-8 -*-
import glob
import io
import os
import setuptools


setuptools.setup(
    name='scopeloompy',
    version='0.0.1',
    description="Python package to create .loom files compatible with SCope",
    python_requires=">=3.5",
    keywords='scope loom single-cell',
    author="Kristofer Davie, Maxime De Waegeneer",
    url='https://github.com/aertslab/SCopeLoomPy.git',
    license='GPL-3.0+',
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[os.path.splitext(os.path.basename(path))[0] for path in glob.glob('src/*.py')],
    include_package_data=True,
)