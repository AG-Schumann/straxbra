import setuptools

with open('requirements.txt') as f:
    requires = [x.strip().split('=')[0] for x in f.readlines()]

setuptools.setup(name='straxbra',
                 version='0.0.1',
                 description='Strax gubbins for XeBRA',
                 author='Darryl Masson',
                 install_requires=requires,
                 python_requires='>=3.7',
                 packages=setuptools.find_packages(),
                 )
