#!/bin/bash
#SBATCH --job-name=datatransfer_test
#SBATCH --output=/home/ofourkioti/Projects/GT-2022-histo/results/datatransfer_test.txt
#SBATCH --error=/home/ofourkioti/Projects/GT-2022-histo/results/datatransfer_test.err
#SBATCH --partition=data-transfer
#SBATCH --ntasks=1
#SBATCH --time=100:00:00

#srun rsync -avP   /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/ovarian_cancer/patches/

#srun rsync -avP /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/
#srun rsync -avP  /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/PDAC_TMA/graphs/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/tmi/PDAC_TMA/graphs/
srun rsync -avP  /data/rds/DBI/DUDBI/DYNCESYS/GOSH/tmi/neuroblastoma/* /data/rds/DBI/DUDBI/DYNCESYS/OlgaF/tmi/neuroblastoma/
