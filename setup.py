# https://pythonhosted.org/an_example_pypi_project/setuptools.html
# set HOME=c:\s\telos\python
# python setup.py sdist bdist_wininst upload.bat
# pip freeze > file to list installed packages
# pip install -r requirements.txt to install

from setuptools import setup
import os

tests_require = ['unittest', 'sly']
install_requires = [
    'aggregate>=0.7.5'
]

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()


long_description = read('README.rst')

setup(name="aggregate_extensions",
      description="aggregate_extensions - applying the aggregate package",
      long_description=long_description,
      license="""BSD""",
      version="0.1.0",
      author="Stephen J. Mildenhall",
      author_email="steve@convexrisk.com",
      maintainer="Stephen J. Mildenhall",
      maintainer_email="steve@convexrisk.com",
      packages=['aggregate_extensions'],
      package_data={'': ['*.txt', '*.rst', '*.md', 'examples/*.py', 'examples/*.ipynb',
                         'test/*.py']},
      tests_require=tests_require,
      install_requires=install_requires,
      classifiers=[
          'Development Status :: 4 - Beta',
          'Programming Language :: Python :: 3',
          'License :: OSI Approved :: BSD License',
          'Topic :: Education',
          'Topic :: Office/Business :: Financial',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Intended Audience :: Financial and Insurance Industry',
          'Intended Audience :: Education'
      ],
      project_urls={"Documentation": 'http://www.mynl.com/aggregate_projectUPDATE/',
                    "Source Code": "https://github.com/mynl/aggregate_projectUPDATE"}
      )
