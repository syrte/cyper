from setuptools import setup

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name='cyper',
    version='1.0',
    description='Cython performed inline: compile and run your Cython snippets on the fly.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/syrte/cyper/',
    keywords="cython",
    author='Syrtis Major',
    author_email='styr.py@gmail.com',
    py_modules=['cyper'],
    install_requires=['Cython'],
    license='MIT License',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Topic :: Software Development :: Build Tools',
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
)
