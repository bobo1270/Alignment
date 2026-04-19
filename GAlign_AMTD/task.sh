#!/bin/bash
#SBATCH -J test 
#SBATCH -p defq
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00

# sleep 10000
# python -u network_alignment.py --source_dataset graph_data/allmv_tmdb/allmv/graphsage \
# --target_dataset graph_data/allmv_tmdb/tmdb/graphsage \
# --groundtruth graph_data/allmv_tmdb/dictionaries/groundtruth GAlign \
# --log --GAlign_epochs 10 --refinement_epochs 0 --cuda

python -u network_alignment.py --source_dataset graph_data/allmv_tmdb2/allmv/graphsage \
--target_dataset graph_data/allmv_tmdb2/tmdb/graphsage \
--groundtruth graph_data/allmv_tmdb2/dictionaries/groundtruth GAlign \
--log --GAlign_epochs 11 --refinement_epochs 0 --cuda

# python -u network_alignment.py --source_dataset graph_data/douban/online/graphsage \
# --target_dataset graph_data/douban/offline/graphsage \
# --groundtruth graph_data/douban/dictionaries/groundtruth GAlign \
# --log --GAlign_epochs 50 --refinement_epochs 0 --cuda \


# python -u network_alignment.py --source_dataset graph_data/douban2/online/graphsage \
# --target_dataset graph_data/douban2/offline/graphsage \
# --groundtruth graph_data/douban2/dictionaries/groundtruth \
# GAlign \
# --log --GAlign_epochs 50 --refinement_epochs 0 --cuda \
# --embedding_dim 128 --lr 0.005 --noise_level 0 --refinement_epochs 0 

# python -u network_alignment.py --source_dataset graph_data/fb_tt/facebook/graphsage \
# --target_dataset graph_data/fb_tt/twitter/graphsage \
# --groundtruth graph_data/fb_tt/dictionaries/groundtruth \
# GAlign \
# --log --GAlign_epochs 50 --refinement_epochs 50 --cuda \


# python -u network_alignment.py --source_dataset graph_data/econ/econ1/graphsage \
# --target_dataset graph_data/econ/econ2/graphsage \
# --groundtruth graph_data/econ/dictionaries/groundtruth \
# GAlign \
# --log --GAlign_epochs 50 --refinement_epochs 0 --cuda \
# --noise_level 0.05

# python -u network_alignment.py --source_dataset graph_data/dblp/dblp1/graphsage \
# --target_dataset graph_data/dblp/dblp1/graphsage \
# --groundtruth graph_data/dblp/dictionaries/groundtruth \
# GAlign \
# --log --GAlign_epochs 50 --refinement_epochs 50 --cuda \
# --noise_level 0.1