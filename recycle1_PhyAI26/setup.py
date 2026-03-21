from setuptools import find_packages, setup

package_name = 'recycle1_PhyAI26'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='TODO',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'move = recycle1_PhyAI26.move:main',
            'detect = recycle1_PhyAI26.detect:main',
        ],
    },
)