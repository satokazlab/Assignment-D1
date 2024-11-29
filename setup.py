from setuptools import find_packages, setup

package_name = 'oakd_chan'

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
    maintainer='suguta',
    maintainer_email='21238284@edu.cc.saga-u.ac.jp',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'oakd_chan_node = oakd_chan.oakd_chan_node:main',
            'alphabet_detection_node = oakd_chan.alphabet_detection_node:main',
            'alphabet_detection_real_node = oakd_chan.alphabet_detection_real_node:main',
            'search_for_box_node = oakd_chan.search_for_box_node:main',
            'search_for_paper_node = oakd_chan.search_for_paper_node:main'
            
        ],
    },
)
