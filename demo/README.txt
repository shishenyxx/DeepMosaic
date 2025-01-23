***
NOTICES:

-Please be aware that the CRAM file line has been omitted in the input.txt as it requires the -C <reference_path> flag when running deepmosaic-draw. The CRAM file was created using hg19, so please use your own downloaded hg19 (or hg38) reference file to test the demo.

-hgdownload from UCSC
https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/

-Currently model training with trainModel.py will only work with the default efficientnet-b4 model. Additional model training will be updated in future release.

-When using the -m flag for deepmosaic-predict to change prediction models, please use the file names exactly as stored within the deepmosaic/models folder and NOT the absolute path as that might cause errors. 
***

ADDITIONS TO DEMO:

A merged bam file of the original 4 sample bams 

"sample_merged.bam"

with the chromosome ID changed from # -> chr# for the header

AND

converted cram file of this merged bam file 
was added into the demo input.txt file. 

The results were separated into: 
"cram_and_merged_result"
"original_bam_samples_result"
in the results folder

----------Edit_Jan162025----------

Hg38 file for merged sample bam, cram, and vcf were added to 
their respective folders for genome build hg38 testing with
CRAM file implementation.

The demo input.txt also received a new line of inputs for
the hg38_cram and hg38_vcf usage.

Hg38_cram_result folder was added to the results folder with
the hg38_cram demo file outputs. 

