from setuptools import find_packages, setup

package_name = 'big_chemistry_scene'

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
    maintainer_email='deltazor@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
		'mini_scene = big_chemistry_scene.mini_scene:main',
		'plan_to_pose = big_chemistry_scene.plan_to_pose:main',
	],
    },
)
