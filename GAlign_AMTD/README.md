# GAlign
Code of the paper: ***[Shaobo Ren, Zhen Liu, Haiyang Ren, Tongle Duan, Shuhang Gu, DDNA: Dual-diffusion-enhanced network alignment, Knowledge-Based Systems, Volume 348, 2026, 116361.]***.

# Environment

* python>=3.5 
* networkx >= 2.4
* pytorch >= 1.2.0 
* numpy >= 1.18.1 

# Running

For allmv_tmdb dataset

```
python -u network_alignment.py --source_dataset graph_data/allmv_tmdb/allmv/graphsage --target_dataset graph_data/allmv_tmdb/tmdb/graphsage --groundtruth graph_data/allmv_tmdb/dictionaries/groundtruth GAlign --log --GAlign_epochs 10 --refinement_epochs 50 --cuda
```

For douban dataset

```
python -u network_alignment.py --source_dataset graph_data/douban/online/graphsage --target_dataset graph_data/douban/offline/graphsage --groundtruth graph_data/douban/dictionaries/groundtruth GAlign --log --GAlign_epochs 50 --refinement_epochs 0 --cuda --embedding_dim 128 --lr 0.005 --noise_level 0 --refinement_epochs 0 
```

# Citation

Please politely cite our work as follows:

*Shaobo Ren, Zhen Liu, Haiyang Ren, Tongle Duan, Shuhang Gu, DDNA: Dual-diffusion-enhanced network alignment, Knowledge-Based Systems, Volume 348, 2026, 116361.*
