from distutils.core import setup

setup(name='mc-providers',
      version='0.1',
      description='A package which contains a nice wrapper for the mediacloud search tools',
      author='mediacloud team',
      author_email='nano3.14@gmail.com',
      packages=['mcproviders',
               'mcproviders.language'],
     )