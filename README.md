# DeepMosaic  <img src="https://user-images.githubusercontent.com/17311837/88461876-52d18f80-ce5c-11ea-9aed-534dfd07d351.png" alt="DeepMosaic_Logo" width=10%> 

Visualization and control-independent classification tool of mosaic SNVs with deep convolutional neural networks.

* Information of aligned sequences for any SNV represented with an RGB image

<img src="https://user-images.githubusercontent.com/17311837/88461918-a04dfc80-ce5c-11ea-888b-4ea8d68d425a.png" alt="Image_representation" width=50%> 

An RGB image was used to represent the pileup results for all the reads aligned to a single genomic position. Reads supporting different alleles were grouped, in the order of the reference allele, the first, second, and third alternative alleles, respectively. Red channel was used to represent the bases, green channel for the base qualitites, and blue channel for the strand orientations of the read.

* DeepMosaic workflow: From variant to result (10 models were compared and Efficientnet b4 was selected as the best-performing): 

     
<img src="https://user-images.githubusercontent.com/17311837/88461821-caeb8580-ce5b-11ea-8c91-7c3ad916fc60.png" alt="DeepMosaic_workflow" width=80%>

Workflow of DeepMosaic on best-performed deep convolutional neural network model after benchmarking. Variants were first transformed into images based on the alignment information. Deep convolution neural network then extract the high-dimontional information from the image, experimental, genomic, and population related information were further incoperated in to the classifier.

--------------------------------------------

## Requirements before you start
* [BEDTools](https://bedtools.readthedocs.io/en/latest/content/tools/coverage.html) (command line)
* [ANNOVAR](https://doc-openbio.readthedocs.io/projects/annovar/en/latest/) (command line)
* [Pysam](https://github.com/pysam-developers/pysam)
* [PyTorch](https://pytorch.org/)
* [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
* [argparse](https://docs.python.org/3/library/argparse.html)

--------------------------------------------

## Getting Started

### Step 0. Installation

0.1. Install DeepMosaic

```
> git clone --recursive https://github.com/Virginiaxu/DeepMosaic
    
> cd DeepMosaic && pip install dist/DeepMosaic-0.0.tar.gz    
```
    
0.2. Install dependency: bedtools (via conda)

```    
> conda install -c bioconda bedtools    
```

0.3. Install dependency: ANNOVAR

   a) Go to the ANNOVAR website and click "[here](http://annovar.openbioinformatics.org/en/latest/user-guide/download/)" to register and download the annovar distribution. 
    
   b) Once you have sucessfully download ANNOVAR package, run
    
```
> cd [path to ANNOVAR]
> perl ./annotate_variation.pl -buildver hg19 -downdb -webfrom annovar gnomad_genome humandb/    
```
    
   to intall the hg19.gnomad_genome file needed for the feature extraction from the bam file
   
0.4. Load bedtools
```
> module load bedtools
```


### Step 1. Image representation of Bam file based on input list of variants and feature extraction:

```
> deepmosaic-draw -i [input.txt] -o [output_dir] -a [path to ANNOVAR] 
```

### Input format

|#sample_name|bam|vcf|depth|sex|
|---|---|---|---|---|
|sample|sample.bam|sample.vcf|200|M|

### Note: sample.vcf is in the format

|#CHROM|POS|ID|REF|ALT|...|
|---|---|---|---|---|---|
|1|17697|.|G|C|.|.|
|1|19890|.|T|C|.|.|


### Step 2. Prediction for mosaicism

```
> deepmosaic-predict -i [output_dir/feature.txt] -o [output.txt] -m [prediction model (default: efficientnet-b4_epoch_6.pt)] -b [batch size (default: 10)]
```

--------------------------------------------


--------------------------------------------

## The intermediate features.txt file

|#sample_name|sex|chrom|pos|ref|alt|variant|maf|lower_CI|upper_CI|variant_type|gene_id|gnomad|all_repeat|segdup|homopolymer|dinucluotide|depth_fraction|image_filepath|npy_filepath|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|


--------------------------------------------
## Output format
|#sample_name|sex|chrom|pos|ref|alt|variant|maf|lower_CI|upper_CI|variant_type|gene_id|gnomad|all_repeat|segdup|homopolymer|dinucluotide|depth_fraction|homo_score|hetero_score|mosaic_score|prediction|image_filepath|


--------------------------------------------

## Demos

--------------------------------------------

## Contact

If you have any questions please contact us:

:email: Xin Xu: [xinxu@hsph.harvard.edu](mailto:xinxu@hsph.harvard.edu), [virginiaxuxin@gmail.com](mailto:virginiaxuxin@gmail.com)

:email: Xiaoxu Yang: [xiy010@health.ucsd.edu](mailto:xiy010@health.ucsd.edu), [yangxiaoxu-shishen@hotmail.com](mailto:yangxiaoxu-shishen@hotmail.com)


