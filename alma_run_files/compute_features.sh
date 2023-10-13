#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=100:00:00
#SBATCH --output=/home/ofourkioti/Projects/GT-2022-histo/results/gtp_feats.out
#SBATCH --error=/home/ofourkioti/Projects/GT-2022-histo/results/error.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuhm

module use /opt/software/easybuild/modules/all/
module load Mamba
source ~/.bashrc
mamba activate  dl_torch
cd /home/ofourkioti/Projects/GT-2022-histo/feature_extractor/

#python build_graphs.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/tiles/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/graphs/"
#python build_graphs.py --weights "model.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/TCGA-LUNG/tiles/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/TCGA-LUNG/graphs"

#python compute_feats_gtp.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/Colonoscopy/colon_patches/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/Colonoscopy/graphs/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/Colonoscopy/segment_dataset/

#python compute_feats_gtp.py --weights "DSMIL_extractors/20x/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/gtp_features/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/images/
python compute_feats_gtp.py --weights "DSMIL_extractors/tcga_lung/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/rcc_graphs/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/TCGA_RCC/
#python compute_feats_res.py --weights "DSMIL_extractors/camelyon/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/sad_mil/feats/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon16/
#python compute_feats_res.py --weights "DSMIL_extractors/tcga_lung/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/rcc_features/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/TCGA_RCC/




#python compute_feats_res.py --weights "DSMIL_extractors/camelyon/model-v0.pth" --dataset "/home/admin_ofourkioti/PycharmProjects/baseline_models/GT-2022-histo/colon_patches/patches/*"  --output colon_feats_SAD --slide_dir /home/admin_ofourkioti/Documents/Colonoscopy/segment_dataset/
