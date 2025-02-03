from setuptools import setup

setup(
   name='diverge-flow',
   version='v0.8.3',
   description='divERGe implements various ERG examples',
   long_description_content_type = 'text/markdown',
   packages=['diverge'],
   package_dir={'diverge': 'util'},
   package_data={'diverge': ['*.so', '*.py']},
   install_requires=['numpy>=1.13', 'setuptools>=30.3.0'],
)

