#run NeuSomatic in a singularity container
#Yuwu Chen from San Diego Supercomputer Center

# clean any existing results 
rm -rf work_standalone
# start a singularity shell to run preprocess.py (note in the singularity shell the command prompt is changed to "Singularity>") 
singularity shell -B /projects,/oasis /projects/neusomatic.simg
Singularity> preprocess.py --mode call --reference /projects/resources/GRCh37/GRCh37.fa --region_bed /projects/resources/b37.bed --tumor_bam /projects/tumor.bam --normal_bam /projects/normal.bam --work work_standalone --scan_maf 0.05 --min_mapq 10 --snp_min_af 0.01 --snp_min_bq 20 --snp_min_ao 10 --ins_min_af 0.05 --del_min_af 0.05 --num_threads 12 --scan_alignments_binary /opt/neusomatic/neusomatic/bin/scan_alignments

singularity exec -B /projects,/oasis /projects/neusomatic.simg call.py --candidates_tsv work_standalone/dataset/*/candidates*.tsv --reference /projects/resources/GRCh37/GRCh37.fa --out work_standalone --checkpoint /projects/neusomatic/neusomatic/models/NeuSomatic_v0.1.4_standalone_SEQC-WGS-GT50-SpikeWGS10.pth --num_threads 12 --batch_size 100

singularity exec -B /projects,/oasis /projects/neusomatic.simg python /projects/neusomatic/neusomatic/python/postprocess.py --reference /projects/resources/GRCh37/GRCh37.fa --tumor_bam
