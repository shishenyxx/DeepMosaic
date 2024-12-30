import sys, os
import pandas as pd
import tempfile
import subprocess
import pkg_resources
import re

#all_repeats_path = pkg_resources.resource_filename('deepmosaic', 'resources/all_repeats.b37.bed')
#segdup_path = pkg_resources.resource_filename('deepmosaic', 'resources/segdup.hg19.bed')
HERE = os.path.abspath(os.path.dirname(__file__))
all_repeats_path = {}
segdup_path = {}
all_repeats_path['hg19'] = os.path.join(HERE, "./resources/all_repeats.b37.bed")
segdup_path['hg19'] = os.path.join(HERE, "./resources/segdup.hg19.bed")
all_repeats_path['hg38'] = os.path.join(HERE, "./resources/all_repeats.b38.bed")
segdup_path['hg38'] = os.path.join(HERE, "./resources/segdup.hg38.bed")

def repeats_annotation(all_variants, output_dir, build):
    rp_fd, rp_path = tempfile.mkstemp()
    #count = 0
    try:
        with os.fdopen(rp_fd, 'w') as tmp:
            # do stuff with temp file
           for variant in all_variants:
               sample_name, bam, chrom, pos, ref, alt, depth, sex = variant
               key = "_".join([chrom, pos, ref, alt])
               line = "\t".join(map(str, [chrom, int(pos)-1, int(pos) + len(ref)-2, ref, alt, key])) + "\n"
               tmp.write(line)
               #count =+ 1
               #if count == 2:
               #break
               #tmp.write("\t".join(map(str, [chrom, int(pos)-1, int(pos) + len(ref)-2, ref, alt, key])) + "\n")

        command = "bedtools annotate -i " + rp_path +" -files " + all_repeats_path[build] + " " +  segdup_path[build] + " > " + \
                   output_dir + "repeats_annotation.bed"
        subprocess.call(command, shell=True)
        os.remove(rp_path)
        df = pd.read_csv(output_dir + "repeats_annotation.bed", header=None, sep="\t", dtype={0: str})
        repeats_dict = dict(zip(df[5], zip(df[6], df[7])))
        return repeats_dict
    except:
        sys.stderr.write("Error with repeat annotation. Check if you have module loaded bedtools.\n")
        sys.exit(2)

