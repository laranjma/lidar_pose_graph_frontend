from setuptools import find_packages, setup

package_name = 'lidar_pose_graph_frontend'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/lidar_pose_graph_demo.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/lidar_pose_graph.rviz']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='matheus',
    maintainer_email='matheus.laranjeira@proton.me',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pose_graph_frontend_backend = lidar_pose_graph_frontend.pose_graph_node:main',
        ],
    },
)
