#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=100:00:00
#SBATCH --output=/home/ofourkioti/Projects/GT-2022-histo/results/train_rcc.out
#SBATCH --error=/home/ofourkioti/Projects/GT-2022-histo/results/train.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
mamba activate  dl_torch
cd /home/ofourkioti/Projects/GT-2022-histo/

#CUDA_VISIBLE_DEVICES=0 python main_2.py --n_class 2 --data_path "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/tcga_lung/" \
#--train_set "TCGA_LUNG_splits/train_3.txt" --val_set "TCGA_LUNG_splits/val_3.txt" --model_path "graph_transformer/saved_models/" \
#--log_path "graph_transformer/runs/" \
#--task_name "tcga_lung_3" \
#--batch_size 4 \
#--train \
#--log_interval_local 5

#for i in {0..4};
#do CUDA_VISIBLE_DEVICES=0 python main.py --n_class 2 --data_path "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/brca/"\
#--train_set "/home/ofourkioti/Projects/HistoTree/brca_files/train_${i}.txt"\
#--val_set "/home/ofourkioti/Projects/HistoTree/brca_files/val_${i}.txt"\
#--model_path "graph_transformer/saved_models/" \
#--log_path "graph_transformer/runs/" \
#--task_name "gtp_brca_${i}"\
#--batch_size 4 \
#--train \
#--log_interval_local 5
#done


for i in {0..4};
do CUDA_VISIBLE_DEVICES=0 python main.py --n_class 3 --data_path "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/" --train_set "/home/ofourkioti/Projects/HistoTree/rcc_files/train_${i}.txt" --val_set "/home/ofourkioti/Projects/HistoTree/rcc_files/val_${i}.txt" --model_path "graph_transformer/saved_models/" --log_path "graph_transformer/runs/" --task_name "gtp_rcc_${i}" --batch_size 4 --train --log_interval_local 5
done