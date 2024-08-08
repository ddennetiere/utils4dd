#!/usr/bin/env python

import setuptools

setuptools.setup(name='utils4dd',
                 version='1.0',
                 description='Python Distribution Utilities',
                 author='David Dennetiere',
                 author_email='david.dennetiere@synchrotron-soleil.fr',
                 requires=['setuptools', 'numpy', 'pandas', 'xarray', 'pytables', 'tkinter', 'plotly', 'scipy'],
                 packages=setuptools.find_packages(),
                 package_data={
                        # all .dat files at any package depth
                        '': ['**/*.dat'],},
                 # data_files=[('utils4dd/data', ['utils4dd/data/f1f2_EPDL97.dat'])],
                 classifiers=[
                     'Development Status :: 4 - Beta',
                     'Environment :: Console',
                     'Environment :: Web Environment',
                     'Intended Audience :: End Users/Desktop',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: Python Software Foundation License',
                     'Operating System :: Microsoft :: Windows',
                     'Programming Language :: Python',
                     'Topic :: Software Development :: Bug Tracking',
                 ],
                 )
