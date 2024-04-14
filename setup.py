import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="timeScaleNetwork",
    version="0.0.1",
    author="trevor",
    author_email="mail2trevorm@gmail.com",
    description="Time Scale Network Pytorch Functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CenterforImplantableDevices/TimeScaleNetwork",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
