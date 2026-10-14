from setuptools import find_packages, setup

package_name = 'big_chemistry_motion'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='deltazor',
    maintainer_email='vetka.adi@gmail.com',
    description='Device-relative motion requests and planning integration for Big Chemistry.',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'target_reader = big_chemistry_motion.target_reader:main',
        ],
    },
)
