try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
setup(
    name='DeepMosaic',
    version='0.0.0',
    author='Xin (Virginia) Xu',
    author_email='xinxu@hsph.harvard.edu',
    description="image based mosaic variant classification tool using CNN based deeplearning",
    packages=['deepmosaic'],
    package_dir={'deepmosaic': 'deepmosaic'},
    package_data={'deepmosaic': ['models/*.pt', 'resources/*']},
    keywords = ['variant-calling', 'cnn', 'deep-learning', "transfer-learning", "mosaic-variant", "bioinformatics-tool"],
    scripts=['deepmosaic/deepmosaic-draw', 'deepmosaic/deepmosaic-predict'],
    url='https://github.com/Virginiaxu/DeepMosaic',
    license='MIT',
    long_description=open('README.md').read(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
	"pysam",
        "efficientnet-pytorch",
    	"argparse",
        "scipy",
        "tables",
        "matplotlib"
    ],
)
