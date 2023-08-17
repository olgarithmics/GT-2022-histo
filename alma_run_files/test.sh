#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=100:00:00
#SBATCH --output=/home/ofourkioti/Projects/GT-2022-histo/results/test_tcga.out
#SBATCH --error=/home/ofourkioti/Projects/GT-2022-histo/results/test_tcga.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
mamba activate  dl_torch
cd /home/ofourkioti/Projects/GT-2022-histo/

for i in {0..3};
do export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 2 \
--data_path '/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/tcga_lung/' \
--val_set "tcga_lung_files/test_${i}.txt" \
--model_path "graph_transformer/saved_models/" \
--log_path "graph_transformer/runs/" \
--task_name "tcga_lung_${i}"  \
--batch_size 1 \
--test \
--log_interval_local 5 \
--resume "graph_transformer/saved_models/tcga_lung_${i}.pth"

done