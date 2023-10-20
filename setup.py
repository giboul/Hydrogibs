from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="hydrogibs",
    version="0.0.62",
    description="A personal hydrology and hydraulics package"
                " based on Christophe Ancey's teaching: "
                "http://fr.ancey.ch/cours/masterGC/cours-hydraulique.pdf",
    packages=find_packages(exclude=["hydrogibs/test*"]),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="giboul",
    author_email="axel.giboulot@epfl.ch",
    license="MIT",
    install_requires=["numpy", "scipy"],
    python_requires=">=3.7",
)
