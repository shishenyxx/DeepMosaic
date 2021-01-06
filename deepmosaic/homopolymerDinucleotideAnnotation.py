from track import Track
import pandas as pd
import tempfile
import pkg_resources
import re
import os.path

#hg19_track_path = pkg_resources.resource_filename('deepmosaic', 'resources/hg19_seq.h5')
#hg38_track_path = pkg_resources.resource_filename('deepmosaic', 'resources/hg38_seq.h5')

# The directory containing this file
HERE = os.path.abspath(os.path.dirname(__file__))
hg19_track_path = os.path.join(HERE, "./resources/hg19_seq.h5")
hg38_track_path = os.path.join(HERE, "./resources/hg38_seq.h5")

def homopolymer_dinucleotide_annotation(chrom, pos, build):
    def check_if_in_homopolymer(seq_str):
        pattern = re.compile(r'([ACGT])\1{3,}')
        matches = [m.group() for m in re.finditer(pattern, seq_str)]
        if len(matches) > 0:
            is_homopolymer = 1
        else:
            is_homopolymer = 0
        return is_homopolymer

    def check_if_in_dinucleotide_repeat(seq_str):
        pattern = re.compile(r'([ACGT]{2})\1{3,}')
        matches = [m.group() for m in re.finditer(pattern, seq_str)]
        if len(matches) > 0:
            is_dinucleotide = 1
        else:
            is_dinucleotide = 0
        return is_dinucleotide

    if build == "hg19":
        track_path = hg19_track_path
    elif build == "hg38":
        track_path = hg38_track_path

    seq_track = Track("seq", track_path)
    pos = int(pos)
    chrom = "chr" + chrom
    seq_str_9bp = seq_track.get_seq_str(chrom, pos-4, pos+4)
    seq_str_17bp = seq_track.get_seq_str(chrom, pos-8, pos+8)
    return check_if_in_homopolymer(seq_str_9bp), check_if_in_dinucleotide_repeat(seq_str_17bp)
