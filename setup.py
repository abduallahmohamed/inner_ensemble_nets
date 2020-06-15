import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ien", # Replace with your own username
    version="0.0.1",
    author="Abduallah Mohamed",
    author_email="abduallah.mohamed@utexas.edu",
    description="Inner Ensemble Net Pytorch Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abduallahmohamed/inner_ensemble_nets",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)