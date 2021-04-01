import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="invertsy",
    version="1.0.0",
    author="Evripidis Gkanias",
    maintainer="Evripidis Gkanias",
    author_email="ev.gkanias@ed.ac.uk",
    maintainer_email="ev.gkanias@ed.ac.uk",
    description="Python package of simulated environments (e.g. sky and the world), using common and easy-to-install"
                "packages, e.g. NumPy and SciPy.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/evgkanias/InvertSy",
    licence="GPLv3+",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Licence :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent"
    ],
    packages=["invertsy"],
    package_dir={"invertsy": "src"},
    python_requires=">=3.8",
)
