import pysam
import random
import pandas as pd
import sys, os
MAX_DP = 500
WIDTH = 300

def construct_reads_dict():
    reads_dict = {}
    reads_dict["A"] = []
    reads_dict["C"] = []
    reads_dict["G"] = []
    reads_dict["T"] = []
    reads_dict["N"] = []
    reads_dict["del"] = []
    return reads_dict


class pysamReader():    
    def __init__(self, file_name, chrom, pos, cram_ref_dir, ref=None, alt=None):
        self.pos = int(pos)-1
        if file_name.endswith('.bam'):
            self.samfile = pysam.AlignmentFile(file_name, "rb")
            self.reads = self.samfile.fetch(chrom, self.pos, self.pos+1)
            if cram_ref_dir != None:
                sys.stdout.write("NOTICE: CRAM reference directory was given when input file type is BAM for " + str(file_name) + "\n")
                #sys.exit(2)
        elif file_name.endswith('.cram'):
            if cram_ref_dir !=  None:
                self.samfile = pysam.AlignmentFile(file_name, "rc", reference_filename=cram_ref_dir)
                self.reads = list(self.samfile.fetch(chrom,self.pos,self.pos+1))
            else: 
                sys.exit("Input is CRAM file but no reference path was given.")
        else:
            sys.stderr.write("Invalid BAM/CRAM type. Make sure the input file is BAM or CRAM.\n")
            sys.exit(2)
        self.reads_count = self.samfile.count(chrom, self.pos, self.pos+1)
        self.reads_dict = construct_reads_dict()
        self.base_info = []
        self.ref = ref
        self.alt = alt
        self.chrom = chrom

    def downsample_to_max_depth(self):
        temp_reads = []
        if self.reads_count > MAX_DP:
            selected_indices = sorted(random.sample(range(self.reads_count), MAX_DP))
            i = 0
            for read in self.reads:
                if i in selected_indices:
                    temp_reads.append(read)
                i += 1  
        else:
            for read in self.reads:
                temp_reads.append(read)
        self.reads = temp_reads

    def build_reads_dict(self):
        #determine which is reference, alt
        for i, read in enumerate(self.reads):
            read_sequence = read.query_sequence
            ref_positions = read.get_reference_positions(full_length=len(read_sequence))
            if self.pos not in ref_positions:
                self.reads_dict["del"].append(read)
            else:
                index = ref_positions.index(self.pos)
                base = read_sequence[index]
                self.reads_dict[base].append(read)
        
    def rearrange_reads_ref_alt(self):
        counts = []
        for base in self.reads_dict.keys():
            if base != self.ref and base != "N" and base != "del":
                counts.append([len(self.reads_dict[base]), base]) 
        counts = pd.DataFrame(counts)
        counts = counts.sort_values([0,1], ascending=[False, True]).values
        alt1 = self.alt
        if counts[0][1] == alt1:
            alt2 = counts[1][1]
            alt3 = counts[2][1]
            count1 = counts[0][0]
            count2 = counts[1][0]
            count3 = counts[2][0]
        elif counts[1][1] == alt1:
            alt2 = counts[0][1]
            alt3 = counts[2][1]
            count1 = counts[1][0]
            count2 = counts[0][0]
            count3 = counts[2][0]
        elif counts[2][1] == alt1:
            alt2 = counts[0][1]
            alt3 = counts[1][1]
            count1 = counts[2][0]
            count2 = counts[0][0]
            count3 = counts[1][0]
        self.reads = self.reads_dict[self.ref] + self.reads_dict[alt1] + \
                     self.reads_dict[alt2] + self.reads_dict[alt3] + \
                     self.reads_dict["N"] + self.reads_dict["del"]
        self.base_info = [[len(self.reads_dict[self.ref]), self.ref], [count1, alt1],\
                 [count2, alt2], [count3, alt3], \
                 [len(self.reads_dict["N"]),"N"], [len(self.reads_dict["del"]), "del"]]

    def rearrange_reads_no_ref(self):
        counts = []
        for base in self.reads_dict.keys():
            if base != "N" and base != "del":
                counts.append([len(self.reads_dict[base]), base])
        counts = sorted(counts, reverse=True)
        base_info = counts + [[len(self.reads_dict["N"]),"N"], [len(self.reads_dict["del"]), "del"]]
        ref = counts[0][1]
        alt1 = counts[1][1]
        alt2 = counts[2][1]
        alt3 = counts[3][1]
        self.reads = self.reads_dict[ref] + self.reads_dict[alt1] + \
                     self.reads_dict[alt2] + self.reads_dict[alt3] + \
                     self.reads_dict["N"] + self.reads_dict["del"]
        self.base_info = counts + [[len(self.reads_dict["N"]),"N"], [len(self.reads_dict["del"]), "del"]]


    def rearrange_reads_no_alt(self):
        counts = []
        for base in self.reads_dict.keys():
            if base != self.ref and base != "N" and base != "del":
                counts.append([len(self.reads_dict[base]), base])
        counts = sorted(counts, reverse=True)
        alt1 = counts[0][1]
        alt2 = counts[1][1]
        alt3 = counts[2][1]
        self.reads = self.reads_dict[self.ref] + self.reads_dict[alt1] + \
                     self.reads_dict[alt2] + self.reads_dict[alt3] + \
                     self.reads_dict["N"] + self.reads_dict["del"]
        self.base_info = counts + [[len(self.reads_dict["N"]),"N"], [len(self.reads_dict["del"]), "del"]]


    def close(self):
        self.samfile.close()
        return [self.reads, self.reads_count, self.base_info]




