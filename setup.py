from setuptools import setup


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="hydrogibs",
    version="0.0.9",
    description="A personal hydrology and hydraulics package"
                " based on Christophe Ancey and Giovanni De Cesare's teaching: "
                "http://fr.ancey.ch/cours/masterGC/cours-hydraulique.pdf",
    packages=["hydrogibs"],
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="giboul",
    author_email="axel.giboulot@epfl.ch",
    license="MIT",
    install_requires=["numpy", "scipy", "matplotlib"],
    python_requires=">=3.7",
)
