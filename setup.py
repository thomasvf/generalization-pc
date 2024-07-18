from setuptools import find_packages, setup

# with open('requirements.txt') as f:
#     required = f.read().splitlines()

setup(
    name='generalization_pc',
    packages=find_packages('src'),
    # install_requires=required,
    package_dir={'': 'src'},
    version='0.0.1',
    description='Package for the pan-cancer generalization experiments.',
    package_data={
        'generalization_pc': ['resources/*'],
    },
)