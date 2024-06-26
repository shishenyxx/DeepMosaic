# Snakemake Wrapper for DeepMosaic

by Jiawei Shen

To facilitate the user to run DeepMosaic on high performance computing clusters at a large scale, we provide some snakemake wrappers for DeepMosaic.

First you should install [Snakemake](https://snakemake.readthedocs.io/en/stable/).

Then you can run Snakemake with `snakemake -j [NUMBER_OF_JOBS] --rerun-incomplete --latency-wait [LATNECY_WAIT_TIME] --cluster “qsub {params.cluster}” --use-conda --conda-frontend conda` in the folder of Snakefile to submit the job and run the pipeline (pbs here used as an example. Switch to sbatch and modify the command accordingly for Slurm).

After evaluation of the average number of input variants, you can choose to download the [split version](https://github.com/Virginiaxu/DeepMosaic/tree/master/Snakemake/Split) or [run everything from a same sample in an individual job version](https://github.com/Virginiaxu/DeepMosaic/tree/master/Snakemake/No_split).
