# DeepMosaic  <img src="https://user-images.githubusercontent.com/17311837/88461876-52d18f80-ce5c-11ea-9aed-534dfd07d351.png" alt="DeepMosaic_Logo" width=15%> 

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

# Installation

### step 1. Install DeepMosaic

```
> git clone --recursive https://github.com/Virginiaxu/DeepMosaic
    
> cd DeepMosaic && pip install dist/DeepMosaic-0.0.tar.gz    
```
    
### step 2. Install dependency: BEDTools (via conda)

```    
> conda install -c bioconda bedtools    
```

### step 3. Install dependency: ANNOVAR

   a) Go to the ANNOVAR website and click "[here](http://annovar.openbioinformatics.org/en/latest/user-guide/download/)" to register and download the annovar distribution. 
    
   b) Once you have sucessfully download ANNOVAR package, run
    
```
> cd [path to ANNOVAR]
> perl ./annotate_variation.pl -buildver hg19 -downdb -webfrom annovar gnomad_genome humandb/    
```
    
   to intall the hg19.gnomad_genome file needed for the feature extraction from the bam file
   
--------------------------------------------

# Usage 

## Step 1. Extract Features and encode image representation of the candidate variants

### Usage

```
> deepmosaic-draw -i [input.txt] -o [output_dir] -a [path to ANNOVAR] 
```
### Note:

1. `[input.txt]` file should be in the following format.

#### Input format

|#sample_name|bam|vcf|depth|sex|
|---|---|---|---|---|
|sample_1|sample_1.bam|sample_1.vcf|200|M|
|sample_2|sample_2.bam|sample_2.vcf|200|F|

Each line of `[input.txt]` is a sample with its aligned reads in the bam format (with index in the same directory), and its candidate variants in the vcf (or vcf.gz) format. User should also provide the sequencing depth and the sex  (M/F) of the corresponding sample. Sample name (#sample_name column) should be a unique identifier for each sample; duplicated names are not allowed.

2. DeepMosaic supports no-loss image representation for sequencing depth up to 500x. Reads with deeper sequencing depth will be randomly down-sampled to 500x during image representation.
 
3. `[sample.vcf]` in each line of the input file should be in the following format:

#### sample.vcf format

|#CHROM|POS|ID|REF|ALT|...|
|---|---|---|---|---|---|
|1|17697|.|G|C|.|.|
|1|19890|.|T|C|.|.|

"#CHROM", "POS", "REF", "ALT" are essential columns that will be parsed and utilized by DeepMosaic.

4. The outputs files including the extracted features and encoded imaged will be output to `[output_dir]`. DeepMosaic will create a new directory if `[output_dir]` hasn't been initialized by users. 

5. `[path to ANNOVAR]` is the absolute path to the ANNOVAR program directory.

### Output:
After deepmosaic-draw is successfully executed, the following files/directories would be generated in the `[output_dir]`

1. `features.txt` contains the extracted features and the absolute path to the encoded image (.npy) file for each variant in each row. `features.txt` will serve as input file to the next step of mosaicism prediction. 

#### features.txt format

|#sample_name|sex|chrom|pos|ref|alt|variant|maf|lower_CI|upper_CI|variant_type|gene_id|gnomad|all_repeat|segdup|homopolymer|dinucluotide|depth_fraction|image_filepath|npy_filepath|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|sample_1| M | 1 | 17697 | G | C | 1_17697_G_C | 0.18236472945891782 | 0.15095348571574527 | 0.21862912439071866 | ncRNA_exonic | WASH7P | 0.1231 | 1 | 1 | 0 | 0 | 3.09 |/.../images/sample_1-1_17697_G_C.jpg | /.../matrices/sample_1-1_17697_G_C.npy|

2. `matrices` is a directory of the encoded image representations in the .npy format for all the candidate variants from all samples. Names of the file would be in the format of `[sample_name]-[chrom]_[pos]_[ref]_[alt].npy`.

2. `images` is a directory of the encoded image representations in the .jpg format for all the candidate variants from all samples. Names of the file would be in the format of `[sample_name]-[chrom]_[pos]_[ref]_[alt].jpg`. Image files in this directory could be directly open and inspected visually by users. 

3. `repeats_annotation.bed` is the intermediate file annotating the repeat and segdup information of each variant.

4. `input.hg19_gnomad_genome_dropped`, `input.hg19_gnomad_genome_filtered`, `input.exonic_variant_function`, `input.variant_function` are ANNOVAR outputs annotating the gnomad and variant function information.

--------------------------------------------

## Step 2. Prediction for mosaicism

### Usage

```
> deepmosaic-predict -i [output_dir/feature.txt] -o [output.txt] -m [prediction model (default: efficientnet-b4_epoch_6.pt)] -b [batch size (default: 10)]
```

### Note:

1. `[output_dir/feature.txt]` is the output file from last step.

2. `[output.txt]` is the final prediction results.

3. `prediction model` is the pretrained DeepMosaic model. The default one (best performing model efficientnet-b4_epoch_6.pt) is trained on our train set for 6 epoch from the efficientnet-b4 architecture.

4. `batch size` is the number of images (variants) predicted by DeepMosaic model simultaneously. Larger batch size means more memory and faster prediction. User can adjust this value depending on his/her available computing power. Default batch size is 10.

### Output:

#### Output format
|#sample_name|sex|chrom|pos|ref|alt|variant|maf|lower_CI|upper_CI|variant_type|gene_id|gnomad|all_repeat|segdup|homopolymer|dinucluotide|depth_fraction|homo_score|hetero_score|mosaic_score|prediction|image_filepath|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|sample_1| M | 1 | 17697 | G | C | 1_17697_G_C | 0.18236472945891782 | 0.15095348571574527 | 0.21862912439071866 | ncRNA_exonic | WASH7P | 0.1231 | 1 | 1 | 0 | 0 | 3.09 |0.9999058880667084 |6.519687262508766e-10 | 9.411128132280348e-05 | artifact| /.../images/sample_1-1_17697_G_C.jpg |

1. The prediction result is in the column "prediction". The possible results are `mosaic`, `heterozygous`, `ref_homozygous`, `alt_homozygous` or `artifact`. Only variants marked by `mosaic` are DeepMosaic predicted mosaic positive. The prediction decision is made by considering the mosaic score generated by DeepMosaic deeplearning model as well as the extracted, user-input, as well as annotated features such as maf, depth_fraction, repeat, segdup, etc. All genomic coordinates and annotations are based on GRCh37d5 reference genome.

2. Image representations of the variants are stored in the files indicated by "image_filepath" column. User can directly open the .jpg files and visually inspect the piled reads for sanity check.

3. Raw extracted, user-input, as well as annotated features are listed in the output file, to allow users to implement further filters: 

`maf`,`lower_CI`, and `upper_CI` are calculated from the mutant allelic fractions and 95% exact binomial confidence intervals extracted from the bam file. 

`variant_type` and `gene_id` are annotated by ANNOVAR.

`gnomad` is annotated from the combined allele frequency in gnomAD (v2.1.1).

`all_repeat` and `segdup` are provided in the "resources" folder.

`homopolymer` and `dinucluotide` are calculated from the .h5 files in the "resources" folder. 


--------------------------------------------


## Contact

If you have any questions please contact us:

[Prof. Joseph Gleeson's lab at UCSD](http://www.gleesonlab.org/index.html)

:email: Prof. Joseph Gleeson (PI): [jogleeson@health.ucsd.edu](mailto:jogleeson@health.ucsd.edu), [contact@gleesonlab.org](mailto:contact@gleesonlab.org) 

 
 
 
:email: Xin Xu: [xinxu@hsph.harvard.edu](mailto:xinxu@hsph.harvard.edu), [virginiaxuxin@gmail.com](mailto:virginiaxuxin@gmail.com)

:email: Xiaoxu Yang: [xiy010@health.ucsd.edu](mailto:xiy010@health.ucsd.edu), [yangxiaoxu-shishen@hotmail.com](mailto:yangxiaoxu-shishen@hotmail.com)


