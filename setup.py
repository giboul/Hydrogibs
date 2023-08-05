from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="hydrogibs",
    version="0.0.32",
    description="A personal hydrology and hydraulics package"
                " based on Christophe Ancey's teaching: "
                "http://fr.ancey.ch/cours/masterGC/cours-hydraulique.pdf",
    # package_dir={"": "hydrogibs"},
    packages=find_packages(),
    data_files=[
        "hydrogibs/floods/GR4.csv",
        "hydrogibs/floods/qdf-mean.csv",
        "hydrogibs/floods/qdf-thres.csv"
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="giboul",
    author_email="axel.giboulot@epfl.ch",
    license="MIT",
    install_requires=["numpy", "scipy"],
    python_requires=">=3",
)
