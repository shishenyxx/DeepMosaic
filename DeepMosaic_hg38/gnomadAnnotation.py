import pandas as pd
import numpy as np
import sys, os
import tempfile
import subprocess
import pkg_resources

def gnomad_annotation(all_variants, output_dir, annovar, annovar_db):
    gm_fd, gm_path = tempfile.mkstemp()
    try:
        with os.fdopen(gm_fd, 'w') as tmp:
            # do stuff with temp file
            for variant in all_variants:
                sample_name, bam, chrom, pos, ref, alt, depth, sex = variant
                key = "_".join([chrom, pos, ref, alt])
                tmp.write("\t".join(map(str, [chrom, int(pos), int(pos) + len(ref) -1, ref, alt, key])) + "\n")
        annovar_command_1 = annovar + " -geneanno -build hg38 -dbtype refGene " + gm_path + " " + annovar_db + " -outfile " + \
                            output_dir + "input"
        subprocess.call(annovar_command_1, shell=True)
        annovar_command_2 = annovar + " -filter -build hg38 -dbtype gnomad_genome " + gm_path + " " + annovar_db + " -outfile " + \
                            output_dir + "input"
        subprocess.call(annovar_command_2, shell=True)
    except:
        sys.stderr.write("Error with gnomad annotation. Check if you have correctly installed Annovar.\n")
        sys.exit(2)
    finally:
        os.remove(gm_path)
    if os.path.exists(output_dir + "input.hg38_gnomad_genome_dropped") and not \
       os.stat(output_dir + "input.hg38_gnomad_genome_dropped").st_size == 0:
            df = pd.read_csv(output_dir + "input.hg38_gnomad_genome_dropped", header=None, sep="\t")
            gnomad_dict = dict(zip(df[7], df[1]))
    else:
        gnomad_dict = {}
    if os.path.exists(output_dir + "input.variant_function") and not \
       os.stat(output_dir + "input.variant_function").st_size == 0:
            df = pd.read_csv(output_dir + "input.variant_function", header=None, sep="\t")
            function_dict = dict(zip(df[7], map(list, zip(df[0], df[1]))))
    else:
        function_dict = {}
    if os.path.exists(output_dir + "input.exonic_variant_function") and not \
       os.stat(output_dir + "input.exonic_variant_function").st_size == 0:
            df = pd.read_csv(output_dir + "input.exonic_variant_function", header=None, sep="\t")
            exonic_dict = dict(zip(df[8], map(list, zip(df[1], df[2]))))
    else:
        exonic_dict = {}
    for key in function_dict.keys():
        if key in exonic_dict.keys():
            function_dict[key][0] += ":" + "_".join(exonic_dict[key][0].split(" "))
            function_dict[key][1] = exonic_dict[key][1]
    return function_dict, gnomad_dict