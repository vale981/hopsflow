from setuptools import setup

setup(
    name="hopsflow",
    version="1.0",
    description="Calculating open system bath energy changes with HOPS and analytically.",
    author="Valentin Boettcher",
    author_email="hiro@protagon.space",
    url="https://github.com/vale981/hopsflow",
    packages=["hopsflow"],
    install_requires=["numpy >= 1.20", "scipy >= 1.6", "h5py", "tqdm", "lmfit"],
)
