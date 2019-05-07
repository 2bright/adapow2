import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="adapow2",
    version="1.0.0",
    author="Liang WenJie",
    author_email="l.wen.jie@qq.com",
    description="Adapow2 is a serial of adaptive gradient descent optimizers by adjusting the power of 2 of a tiny step size.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/2bright/adapow2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache-2.0",
    packages=setuptools.find_packages(),
)
