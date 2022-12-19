import torch
import numpy as np
import json


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, root, half_res=False, testskip=1):
    splits = ['test']
    metas = {}
    for s in splits:
        with open(root, 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        # imgs = []
        poses = []
        skip = 1

        for frame in meta['frames'][::skip]:
            all_imgs.append(frame['file_path'])
            poses.append(np.array(frame['transform_matrix']))
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_poses.append(poses)

    num_selection = 1
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(num_selection)]

    poses = np.concatenate(all_poses, 0)

    # H, W = imgs[0].shape[:2]
    H, W = (800, 800)
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

    return all_imgs, poses, render_poses, [H, W, focal], i_split


