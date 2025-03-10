from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import os
import numpy as np
# from ripser import ripser
# from persim import plot_diagrams
from pointcept.engines.defaults import default_config_parser, default_setup
from gpu_main import load_weights_ptv3_nucscenes_seg, PRETRAINED_PATH
from preprocess_mini_nuscenes import GetMiniNusceneDataset
import matplotlib.pyplot as plt
from pointcept.utils.env import get_random_seed, set_seed
from pointcept.utils.config import Config
import pointcept.utils.comm as comm
import ast


def test_topo(index=20):
    nusc = NuScenes(version='v1.0-mini', dataroot='/home/luutunghai@gmail.com/projects/PTv3-distill/mini_dataset',
                    verbose=True)

    my_sample = nusc.sample[index]

    sample_data_token = my_sample['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)

    sample_rec = nusc.get('sample', sd_record['sample_token'])

    chan = sd_record['channel']
    ref_chan = 'LIDAR_TOP'
    ref_sd_token = sample_rec['data'][ref_chan]
    ref_sd_record = nusc.get('sample_data', ref_sd_token)

    pcl_path = os.path.join(nusc.dataroot, ref_sd_record['filename'])

    pc = LidarPointCloud.from_file(pcl_path)

    point_clouds = pc.points

    # print(my_sample['token'])
    # print(sample_rec)
    # print(pc.points)
    # print(pc.points.shape)
    print(point_clouds)


def default_config_parser_mini(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1:]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    # os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    # if not cfg.resume:
    #     cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup_mini(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )
    # update data loop
    assert cfg.epoch % cfg.eval_epoch == 0
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed * cfg.num_worker_per_gpu + rank
    set_seed(seed)
    return cfg


def test_get_features(index=20):
    from gpu_main import get_teacher_model

    data_root = '/home/luutunghai@gmail.com/projects/PTv3-distill/mini_dataset'
    CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-mini-ver.py"
    # Load configuration
    cfg = default_config_parser_mini(CONFIG_FILE, None)
    cfg = default_setup_mini(cfg)
    data_cfg = cfg.data.mini.transform

    print("------> Loading mini dataset")
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=False)

    preprare_mini_nuscene = GetMiniNusceneDataset(nusc=nusc,
                                                  data_root=data_root,
                                                  split="mini", ignore_index=-1,
                                                  device='cuda',
                                                  cfg=data_cfg)

    mini_data_dict = preprare_mini_nuscene.get_mini_dataset(index)

    print(mini_data_dict)
    print(" ")

    print("------> Loading teacher model")
    # The data needs to process in a dict, so do it later
    # ---------- Load Model -----------------
    CONFIG_FILE = "configs/nuscenes/semseg-pt-v3m1-0-base.py"

    cfg = default_config_parser(CONFIG_FILE, None)
    cfg = default_setup(cfg)

    # Load teacher model
    teacher_model = get_teacher_model(cfg)

    # Load pretrained weights
    teacher_model = load_weights_ptv3_nucscenes_seg(teacher_model, PRETRAINED_PATH).to('cuda')
    # -----------------------------------------
    print(" ")
    print("------> Calculating features")
    teacher_latent_feature = teacher_model.viz_persistent_homology(mini_data_dict)

    print(teacher_latent_feature)


def test_topo_layer():
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import ripserplusplus as rpp_py
    from ripser import ripser
    from persim import plot_diagrams
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from scipy.spatial import Delaunay
    from gpu_main import normalize_diagram

    # torch layer
    import torch_tda
    import bats

    flags = (bats.standard_reduction_flag(), bats.compression_flag())
    layer = torch_tda.nn.RipsLayer(maxdim=1, reduction_flags=flags, sparse=True)

    # Load the .npy file
    TDA_feature = np.load("./TDA_latent_feature.npy", allow_pickle=True)
    point_feature = np.load("./point_feat.npy", allow_pickle=True)

    # Print the shape and type of the data
    print("Data Shape:", point_feature.shape)
    print("Data Type:", type(point_feature))
    # print("Data example: ", point_feature[0])

    latent_space = np.rot90(point_feature)

    # # Reduce dimensionality using PCA (to 50D for efficiency)
    # pca = PCA(n_components=10)
    # latent_reduced = pca.fit_transform(latent_space)

    # Compute persistent homology on the reduced data
    diagrams = ripser(latent_space, metric="euclidean", maxdim=2)['dgms']

    # Convert diagrams to PyTorch tensors
    diagrams_tensors = [
        torch.tensor(dgm, dtype=torch.float32).cuda()
        for dgm in diagrams
    ]

    # Filter out infinite values (optional, recommended)
    diagrams_tensors = [
        dgm[torch.isfinite(dgm).all(dim=1)]
        for dgm in diagrams_tensors
    ]

    # pe_latent_space = rpp_py.run("--format point-cloud --dim 2 --sparse", latent_space)

    print("latent space by ripser")
    print(diagrams_tensors[0])
    print(diagrams_tensors[1])
    print(" ")
    print("latent space by torch layer")
    pca = PCA(n_components=10)
    latent_reduced = pca.fit_transform(latent_space)

    dgms = layer(torch.from_numpy(latent_space.copy()))  # latent_space.copy()
    print(dgms[0])

    norm_diagram = normalize_diagram(diagrams_tensors[0])
    print("Norm the latent space")
    print(norm_diagram)


# test_get_features()
test_topo_layer()
