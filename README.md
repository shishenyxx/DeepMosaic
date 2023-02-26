# DeepMosaic  <img src="https://user-images.githubusercontent.com/17311837/88461876-52d18f80-ce5c-11ea-9aed-534dfd07d351.png" alt="DeepMosaic_Logo" width=15%> 

Visualization and control-independent classification tool of noncancer (somatic or germline) mosaic single nucleotide variants (SNVs) with deep convolutional neural networks.

--------------------------------------------

# Contents

[Overview](#Overview)

[Requirements before you start](#Requirements-before-you-start)

[Installation](#Installation)

[Usage](#Usage)

[-Step 1. Feature extraction and visualization of the candidate mosaic variants(DeepMosaic Visualization Module)](#step-1-feature-extraction-and-visualization-of-the-candidate-mosaic-variants-visualization-module)

[-Step 2. Prediction for mosaicism (DeepMosaic Classification Module)](#step-2-prediction-for-mosaicism-classification-module)

[Demo](#Demo)

[Model Training](#model-training)

[Performance](#Performance)

[Q&A](#qa)

[Cite DeepMosaic](#cite-deepmosaic)

[Licence](#licence)

[Contact](#Contact)

--------------------------------------------

# Overview


* <b>DeepMosaic Visualization Module:</b> Information of aligned sequences for any SNV represented with an RGB image:

<img src="https://user-images.githubusercontent.com/17311837/95255932-fb925880-07d6-11eb-8c16-7d2a5e1d12ed.png" alt="Image_representation" width=50%> 

An RGB image was used to represent the pileup results for all the reads aligned to a single genomic position. Reads supporting different alleles were grouped, in the order of the reference allele, the first, second, and third alternative alleles, respectively. Red channel was used to represent the bases, green channel for the base qualitites, and blue channel for the strand orientations of the read. Note that the green channel is modified to show better contrast for human eyes.

* <b>DeepMosaic Classification Module:</b> Workflow from variant to result (10 models were compared and Efficientnet b4 was selected as default because it performed the best on a gold standard benchmark dataset.): 

     
<img src="https://user-images.githubusercontent.com/17311837/95255993-0fd65580-07d7-11eb-843d-2a3950469cb9.png"  width=80%>

Workflow of DeepMosaic on best-performed deep convolutional neural network model after benchmarking. Variants were first transformed into images based on the alignment information. Deep convolution neural network then extract the high-dimontional information from the image, experimental, genomic, and population related information were further incoperated in to the classifier.

[Return to Contents](#Contents)

--------------------------------------------

<details><summary>

# Requirements before you start 

</summary>

* [git-lfs](https://github.com/git-lfs/git-lfs) for the system you work on
* [BEDTools](https://bedtools.readthedocs.io/en/latest/content/tools/coverage.html) (command line)
* [ANNOVAR](https://doc-openbio.readthedocs.io/projects/annovar/en/latest/) (command line)
* [PyTables](https://pypi.org/project/tables/)
* [Matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)
* [Pysam](https://github.com/pysam-developers/pysam)
* [PyTorch](https://pytorch.org/) version>=1.6.0
* [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) version>=0.7.1
* [argparse](https://docs.python.org/3/library/argparse.html)

[Return to Contents](#Contents)

</details>

--------------------------------------------

<details><summary>

# Installation 

</summary>

## Step 1. Install DeepMosaic

Make sure you have <b>git-lfs</b> installed in your environment (download [git-lfs](https://github.com/git-lfs/git-lfs/releases/), unzip the tar.gz and put the binary file ```git-lfs``` in your bin folder/your $PATH, and run ```git lfs install``` to initialize git-lfs, you only need to do it once) to be able to download this repository correctly. 

```
> git clone --recursive https://github.com/Virginiaxu/DeepMosaic
```

Make sure you cloned the whole repository, total folder size should be ~ 4G.

```    
> cd DeepMosaic   
```
    
## Step 2. Install dependency: BEDTools (via conda)

```    
> conda install -c bioconda bedtools    
```

## Step 3. Install dependency: ANNOVAR

   a) Go to the ANNOVAR website and click "[here](http://annovar.openbioinformatics.org/en/latest/user-guide/download/)" to register and download the annovar distribution. 
    
   b) Once you have sucessfully download ANNOVAR package, run
    
```
> cd [path to ANNOVAR]

> perl ./annotate_variation.pl -buildver hg19 -downdb -webfrom annovar gnomad_genome humandb/    
```
    
   to intall the hg19.gnomad_genome file needed for the feature extraction from the bam file

[Return to Contents](#Contents)

</details>

--------------------------------------------



# Usage 

<details><summary>

## Step 1. Feature extraction and visualization of the candidate mosaic variants (Visualization Module)

</summary>
     
This step is used for the extraction of genomic features of the variant from raw bams as well as population information. It can serve as an independent tool for the visualization and evaluation of mosaic candidates.

### Usage

```
> [DeepMosaic Path]/deepmosaic/deepmosaic-draw -i [input.txt] -o [output_dir] -a [path to ANNOVAR] 
```
### Note:

1. `[input.txt]` file should be in the following format.

#### Input format

|#sample_name|bam|vcf|depth|sex|
|---|---|---|---|---|
|sample_1|sample_1.bam|sample_1.vcf|200|M|
|sample_2|sample_2.bam|sample_2.vcf|200|F|

Each line of `[input.txt]` is a sample with its aligned reads in the bam format (with index in the same directory), and its candidate variants in the vcf (or vcf.gz) format. User should also provide the sequencing depth and the sex  (M/F) of the corresponding sample. Sample name (#sample_name column) should be a unique identifier for each sample; duplicated names are not allowed.

Note the sequencing depth is required for increasing specificity and if the user is not clear about the average depth, we recommend piloting a fast depth analysis with SAMtools mpileup for several hundreds of variants, or a complete depth of coverage analysis. The depth value should be integers. 

2. DeepMosaic supports no-loss image representation for sequencing depth up to 500x. Reads with deeper sequencing depth will be randomly down-sampled to 500x during image representation.
 
3. `[sample.vcf]` is the vcf file of input variants you are interested in, or prior file generated by [GATK haplotypecaller](https://gatk.broadinstitute.org/hc/en-us/articles/360037225632-HaplotypeCaller) with polidy 50 as [described in previosu pipelines](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02285-3), or [MuTect2 single mode](https://gatk.broadinstitute.org/hc/en-us/articles/360037593851-Mutect2), each vcf should be provided for each input bam and the format should be in the following format, gziped vcf is also recognizable:

#### sample.vcf format

|#CHROM|POS|ID|REF|ALT|...|
|---|---|---|---|---|---|
|1|17697|.|G|C|.|.|
|1|19890|.|T|C|.|.|

"#CHROM", "POS", "REF", "ALT" are essential columns that will be parsed and utilized by DeepMosaic.

While using MuTect2 we recommend "PASS" vcfs as input for DeepMosaic. Running MuTect2 single mode, [generate the panel of normals](https://gatk.broadinstitute.org/hc/en-us/articles/360037227652-CreateSomaticPanelOfNormals-BETA-) and downstream filtering could either be found following the [official GATK tutorials](https://gatk.broadinstitute.org/hc/en-us/articles/360037593851-Mutect2), or following [this example snakemake pipeline](https://github.com/shishenyxx/Adult_brain_somatic_mosaicism/tree/master/pipelines/WGS_SNV_indel_calling_pipeline/Mutect2_single_mode).

4. The outputs files including the extracted features and encoded imaged will be output to `[output_dir]`. DeepMosaic will create a new directory if `[output_dir]` hasn't been initialized by users. 

5. `[path to ANNOVAR]` is the absolute path to the ANNOVAR program directory.

### Output:
After deepmosaic-draw is successfully executed, the following files/directories would be generated in the `[output_dir]`

1. `features.txt` contains the extracted features and the absolute path to the encoded image (.npy) file for each variant in each row. `features.txt` will serve as input file to the next step of mosaicism prediction. 

#### features.txt format

|#sample_name|sex|chrom|pos|ref|alt|variant|maf|lower_CI|upper_CI|variant_type|gene_id|gnomad|all_repeat|segdup|homopolymer|dinucleotide|depth_fraction|image_filepath|npy_filepath|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|sample_1| M | 1 | 17697 | G | C | 1_17697_G_C | 0.18236472945891782 | 0.15095348571574527 | 0.21862912439071866 | ncRNA_exonic | WASH7P | 0.1231 | 1 | 1 | 0 | 0 | 3.09 |/.../images/sample_1-1_17697_G_C.jpg | /.../matrices/sample_1-1_17697_G_C.npy|

2. `matrices` is a directory of the encoded image representations in the .npy format for all the candidate variants from all samples. Names of the file would be in the format of `[sample_name]-[chrom]_[pos]_[ref]_[alt].npy`.

2. `images` is a directory of the encoded image representations in the .jpg format for all the candidate variants from all samples. Names of the file would be in the format of `[sample_name]-[chrom]_[pos]_[ref]_[alt].jpg`. Image files in this directory could be directly open and inspected visually by users. 

3. `repeats_annotation.bed` is the intermediate file annotating the repeat and segdup information of each variant.

4. `input.hg19_gnomad_genome_dropped`, `input.hg19_gnomad_genome_filtered`, `input.exonic_variant_function`, `input.variant_function` are ANNOVAR outputs annotating the gnomad and variant function information.

[Return to Contents](#Contents)

</details>

--------------------------------------------

<details><summary>

## Step 2. Prediction for mosaicism (Classification Module)

</summary>
     
### Usage

```
> [DeepMosaic Path]/deepmosaic/deepmosaic-predict -i [output_dir/feature.txt] -o [output.txt] -m [prediction model (default: efficientnet-b4_epoch_6.pt)] -b [batch size (default: 10)]
```

### Note:

1. `[output_dir/feature.txt]` is the output file from last step.

2. `[output.txt]` is the final prediction results.

3. `prediction model` is the pretrained DeepMosaic model. The default one (best performing model efficientnet-b4_epoch_6.pt) is trained on our train set for 6 epoch from the efficientnet-b4 architecture.

4. `batch size` is the number of images (variants) predicted by DeepMosaic model simultaneously. Larger batch size means more memory and faster prediction. User can adjust this value depending on his/her available computing power. Default batch size is 10.

### Output:

#### Output format
|#sample_name|sex|chrom|pos|ref|alt|variant|maf|lower_CI|upper_CI|variant_type|gene_id|gnomad|all_repeat|segdup|homopolymer|dinucleotide|depth_fraction|score1|score2|score3|prediction|image_filepath|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|sample_1| M | 1 | 17697 | G | C | 1_17697_G_C | 0.18236472945891782 | 0.15095348571574527 | 0.21862912439071866 | ncRNA_exonic | WASH7P | 0.1231 | 1 | 1 | 0 | 0 | 3.09 |0.9999058880667084 |6.519687262508766e-10 | 9.411128132280348e-05 | artifact| /.../images/sample_1-1_17697_G_C.jpg |

1. The prediction result is in the column "prediction". The possible results are `mosaic`, `heterozygous`, `ref_homozygous`, `alt_homozygous` or `artifact`. Only variants marked by `mosaic` are DeepMosaic predicted mosaic positive. The prediction decision is made by considering the mosaic score generated by DeepMosaic deeplearning model as well as the extracted, user-input, as well as annotated features such as maf, depth_fraction, repeat, segdup, etc. All genomic coordinates and annotations are based on GRCh37d5 reference genome.

2. Image representations of the variants are stored in the files indicated by "image_filepath" column. User can directly open the .jpg files and visually inspect the piled reads for sanity check.

3. Raw extracted, user-input, as well as annotated features are listed in the output file, to allow users to implement further filters: 

`maf`,`lower_CI`, and `upper_CI` are calculated from the mutant allelic fractions and 95% exact binomial confidence intervals extracted from the bam file. 

`variant_type` and `gene_id` are annotated by ANNOVAR.

`gnomad` is annotated from the combined allele frequency in gnomAD (v2.1.1).

`all_repeat` and `segdup` are provided in the "resources" folder.

`homopolymer` and `dinucleotide` are calculated from the .h5 files in the "resources" folder. 

## We also provided a [Snakemake wrapper](https://github.com/Virginiaxu/DeepMosaic/tree/master/Snakemake) for DeepMosaic users.

[Return to Contents](#Contents)

</details>

--------------------------------------------

<details><summary>

# Demo 

</summary>

We have provided a simple example in the sub-directory of "demo". The directory includes the input files and the expected results from running DeepMosaic. User could refer to the example for the expected input format and output format.

## "Demo" Directory hierarchy
##### --input.txt
##### ---vcfs
     sample_1.vcf
     sample_2.vcf
     sample_3.vcf
     sample_4.vcf
##### ---bams
     sample_1.bam  sample_1.bam.bai
     sample_2.bam  sample_2.bam.bai
     sample_3.bam  sample_3.bam.bai
     sample_4.bam  sample_4.bam.bai
##### ---results
     features.txt                (intermediate result of running deepmosaic-draw)
     final_predictions.txt       (final result of running deepmosaic-predict)
     -----images (image encodings in .jpg formats)
     -----matrices (image encodings in .npy format to be used in prediction directly)
     repeat.annotation.bed       (intermediate file for repeat annotation)
     input.variant_function, input.exonic_variant_function, input.hg19_gnomad_genome_dropped, input.hg19_gnomad_genome_filtered, input.log (intermediate files after running annovar)


#### Demo input: input.txt

|#sample_name|bam|vcf|depth|sex|
|---|---|---|---|---|
|sample_1|bams/sample_1.bam|vcfs/sample_1.vcf|200|M|
|sample_2|bams/sample_2.bam|vcfs/sample_2.vcf|200|M|
|sample_3|bams/sample_3.bam|vcfs/sample_3.vcf|200|M|
|sample_4|bams/sample_4.bam|vcfs/sample_4.vcf|200|M|


#### Expected output: results/final_predictions.txt

|#sample_name|sex|chrom|pos|ref|alt|variant|maf|lower_CI|upper_CI|variant_type|gene_id|gnomad|all_repeat|segdup|homopolymer|dinucluotide|depth_fraction|score1|score2|score3|prediction|image_filepath|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|sample_1| M |      10  |    25509499   |     A   |    G   |    10_25509499_A_G |0.05737704918032788  |   0.03448247887605271   |  0.09399263167327017  |   intronic  |      GPR158 | 0.0  |   0    |   0   |    1   |    0  |     1.22  |  0.00010761513038663674 | 3.852715883900453e-05 |  0.9998538577107744   |   mosaic | results/images/sample_1-10_25509499_A_G.jpg|
|sample_2   |     M      | 14   |   37531674     |   A   |    T  |     14_37531674_A_T| 0.9948186528497408 |     0.9712392635106 |0.9990847787125622   |   intronic|        SLC25A21   |     0.2267 | 1    |   0    |   1   |    1   |    0.98  |  0.19976294102631714  |   4.0270887857736005e-06|  0.800233031884897   |    alternative_homozygous  |results/images/sample_2-14_37531674_A_T.jpg|
|sample_3   |     M    |   20   |   1805075| G   |    T   |    20_1805075_G_T |  0.018072289156626502  |  0.008308354195089811  |  0.03886110152464575   |  intergenic |     LOC100289473(dist=44683),SIRPA(dist=69738) |     0.0    | 0    |   0   |    0   |    0   |    1.66  |  0.003562673370702711    |2.9057256040721804e-06 | 0.9964344209036933  |    mosaic  | results/images/sample_3-20_1805075_G_T.jpg|
|sample_4   |     M    |   16  |    65589896   |     G    |   C   |    16_65589896_G_C |0.5306122448979592  |    0.43252467204457545  |   0.6263904306010359    |  ncRNA_intronic|  LINC00922   |    0.3142 | 1   |    0 |      1  |     0  |     0.49 |   0.9998079754132149 |     5.6467567415316954e-08 | 0.00019196811921752858  |heterozygous |   results/images/sample_4-16_65589896_G_C.jpg|

[Return to Contents](#Contents)

</details>

--------------------------------------------

<details><summary>

# Model Training

</summary>

If you have you own training set, you can train you own DeepMosaic model using [trainModel.py](https://github.com/Virginiaxu/DeepMosaic/blob/master/deepmosaic/trainModel.py). 

-i: input file, tab delimiated |path_to_npy_file_generated_by_DeepMosaic_draw|label|

-e: training epoches

-o: output directory

--model_type: supported model types, see the model folder

--model_path: path to the base model (pt file)

example command:

`python trainModel.py -i test_input_training_10.csv -e 2 --model_type efficientnet-b4 --model_path efficientnet-b4_epoch_6.pt -o ./test_trained_model`


[Return to Contents](#Contents)

</details>

--------------------------------------------

<details><summary>

# Performance

</summary>

1. WGS

We estimated > 90% experimental validation rate for WGS data identified as "mosaic" by DeepMosaic (GRCh37).

2. WES

We estimated ~40% experimental validation rate for WES data identified as "mosaic" by the current DeepMosaic WGS model (GRCh37).

Note that the performance of DeepMosaic on GRCh38 might be different.

[Return to Contents](#Contents)

</details>

--------------------------------------------

<details><summary>

# Q&A

</summary>

## [Starting from Jan 2023, new Q&A section will be added to the wiki page](https://github.com/Virginiaxu/DeepMosaic/wiki), please also visit the [issues](https://github.com/Virginiaxu/DeepMosaic/issues) section so see whether other users already encountered the same questions.

1. Q: How do I run DeepMosaic for multiple samples most efficiently?
   
   A: If you have a large number of variants in each file, to run DeepMosaic in parallel, submit each file in independent input files. If you have a relatively small number of variants from each file but multiple files (samples), integrate everything together into one input file. If you have a huge vcf, you can split it into smaller vcfs and run them parallelly (for both visualization and quantification). You only need to split the vcf, not the bam file.
   
   
2. Q: How do I balance/further filter the variants base on DeepMosaic output?
   
   A: For WGS variants, the exclusion of annotated homopolymer and dinucleotide repeats will remove false positives and increase the validation rate, but decrease the sensitivity.
   
   
3. Q: What do Score 1, Score 2, and Score 3 mean in the output file?
   
   A: The three scores are combined information from the complex features extracted by the neural network, from our experiences, Score 1 is more like a "het and homo probability", Scores 2&3, especially Score 3 is more like a "potential mosaic possibility". In other words, the higher Score 1 is, the more likely the candidate is a germline variant, whereas the higher Score 3 is, the more likely the candidate is a mosaic variant. But both categories contained a lot of potential artifacts, that's why for the final output we included a more complex classifier.


4. Q: How to deal with mitochondria and sex chromosomes?
   
   A: First you should choose a reference genome that supports mitochondria as a separate chromosome. DeepMosaic is not specifically trained on mitochondria variants so we can't guarantee the result, thus we suggest removing the MT variants from DeepMosaic input. For sex chromosomes, DeepMosaic takes into consideration the biological gender of the input sample and also considered the pseudo autosomal regions separately.

5. Q: Can I use DeepMosaic for cancer somatic mutation detection without control?
     
   A: The current models presented by DeepMosaic does not support cancer samples, according to benchmarks, the specificity is high (0.97) while the sensitivity is low. We are training new models that support single sample accurate detection of somatic mutations in cancer.


6. Q: What genome versions does DeepMosaic support?

   A: DeepMosaic is benchmarked on GRCh37(hg19) we are working on some tests for GRCh38(hg39) and are providing some scripts [here](https://github.com/Virginiaxu/DeepMosaic/tree/master/DeepMosaic_hg38) the model is still the same so the main differences lie in coordinate differences. We will make further updates when we finish new models trained on GRCh38 or CHM13. As most of our current benchmark experiments are carried out on GRCh37 we cannot guarantee the performance on GRCh38.
   
7. Q: Why I got errors about pickle_module.load(f, **pickle_load_args)?

   A: Because you didn't fully download DeepMosaic, the entire model folder should be more than 200 MB. Please refer to the git-lfs section in the tutorial.
   
[Return to Contents](#Contents)

</details>

--------------------------------------------

# Cite DeepMosaic
Yang X*<sup>,#</sup>, Xin X*, <i>et al.</i> Gleeson JG<sup>#</sup>. Control-independent mosaic single nucleotide variant detection with DeepMosaic. ([<i>Nature Biotechnology</i>](https://www.nature.com/articles/s41587-022-01559-w))

The Manuscript is also available [here](https://github.com/Virginiaxu/DeepMosaic/blob/master/Publication/s41587-022-01559-w.pdf).

--------------------------------------------

# Licence
Released under GNU-GPL 3.0 [licence](https://github.com/Virginiaxu/DeepMosaic/blob/master/LICENSE).

--------------------------------------------

# Contact

If you have any questions please post a thread at the [issues](https://github.com/Virginiaxu/DeepMosaic/issues) section or contact us at:

:email: Xiaoxu Yang: [xiy010@health.ucsd.edu](mailto:xiy010@health.ucsd.edu), [yangxiaoxu-shishen@hotmail.com](mailto:yangxiaoxu-shishen@hotmail.com)

:email: Xin Xu: [virginiaxuxin@gmail.com](mailto:virginiaxuxin@gmail.com)

:email: Joseph Gleeson: [jogleeson@health.ucsd.edu](mailto:jogleeson@health.ucsd.edu), or [visit the Gleeson lab](https://gleesonlab.ucsd.edu) 

[Return to Contents](#Contents)

