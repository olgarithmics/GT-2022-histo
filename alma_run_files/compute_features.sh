#!/bin/bash
#SBATCH --job-name=PatchExtractor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=76:00:00
#SBATCH --output=/home/ofourkioti/Projects/GT-2022-histo/results/rcc_feats.out
#SBATCH --error=/home/ofourkioti/Projects/GT-2022-histo/alma_run_files/error.err
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
#python compute_feats_gtp.py --weights "DSMIL_extractors/camelyon/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/graphs/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/
#python compute_feats_res.py --weights "DSMIL_extractors/camelyon/model-v0.pth"  --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/SAR/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/SAR/feats/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/SAR/

python compute_feats_res.py --weights "runs/tcga_rcc/checkpoints/rcc_model.pth"  --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/feats/" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/TCGA_RCC/

#python compute_feats_gtp.py --weights "runs/neuroblastoma/checkpoints/neuroblastoma_model.pth"  --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/neuroblastoma/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/neuroblastoma/graphs" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/GOSH/simclr_patches/single_neuroblastoma/
#
#python compute_feats_gtp.py --weights "DSMIL_extractors/camelyon/model-v0.pth" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/patches/*"  --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/rcc/graphs" --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/TCGA_RCC/
#python compute_feats_dsmil.py --weights_low "low_high_mag_embedders_camleyon/5x/" --batch_size 512 --weights_high "low_high_mag_embedders_camleyon/20x/" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/magnification_5x/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-16/low_high_feats/" --magnification tree --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon16/
#python compute_feats_dsmil.py --weights_low "TCGA_feature_extractor/weights-low-mag/" --batch_size 512 --weights_high "TCGA_feature_extractor/" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/tcga_lung/magnification_5x/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/tcga_lung/low_high_feats_level_0/" --magnification tree --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/TCGA_LUNG/TCGA_flat/
#python compute_feats_dsmil.py --weights_low "low_high_mag_embedders_camleyon/5x/" --batch_size 512 --weights_high "low_high_mag_embedders_camleyon/20x/" --dataset "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/magnification_5x/patches/*" --output "/data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/tmi/cam-17/low_high_feats/" --magnification tree --slide_dir /data/scratch/DBI/DUDBI/DYNCESYS/OlgaF/slides/camelyon17/--