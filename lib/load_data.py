import numpy as np

from .load_llff import load_llff_data
from .load_blender import load_blender_data
from .load_nsvf import load_nsvf_data
from .load_blendedmvs import load_blendedmvs_data
from .load_tankstemple import load_tankstemple_data
from .load_deepvoxels import load_dv_data
from .load_co3d import load_co3d_data
from .load_nerfpp import load_nerfpp_data


def load_data(args, arg2):

    K, depths = None, None
    near_clip = None

    images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, arg2.json, args.half_res, args.testskip)
    print('Loaded blender', render_poses.shape, hwf, args.datadir)
    i_test = i_split

    near, far = 2., 6.
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    HW = np.array([(H, W) for im in images])
    # irregular_shape = (images.dtype is np.dtype('object'))

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if len(K.shape) == 2:
        Ks = K[None].repeat(len(poses), axis=0)
    else:
        Ks = K

    render_poses = render_poses[...,:4]

    data_dict = dict(
        hwf=hwf, HW=HW, Ks=Ks,
        near=near, far=far, near_clip=near_clip,
        i_test=i_test,
        poses=poses, render_poses=render_poses,
        images=images, depths=depths,

    )
    return data_dict


def inward_nearfar_heuristic(cam_o, ratio=0.05):
    dist = np.linalg.norm(cam_o[:,None] - cam_o, axis=-1)
    far = dist.max()  # could be too small to exist the scene bbox
                      # it is only used to determined scene bbox
                      # lib/dvgo use 1e9 as far
    near = far * ratio
    return near, far

