from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import argparse

import torch

import numpy as np
from tqdm import tqdm

import trimesh
import smplx
from detect_inter_collisions import numpy2set, detect_collision


def get_smplx_mesh(fn):
    # read smplx params from file
    data = np.load(fn, allow_pickle=True).item()
    transl = torch.from_numpy(data['transl']).float()
    betas = torch.from_numpy(data['betas']).float()
    global_orient = data['global_orient']
    poses = data['poses']
    gender = data['meta']['gender']
    nframe, num_betas = betas.shape

    assert (global_orient == poses[:, :3]).all()
    poses = torch.from_numpy(poses).float().reshape(-1, 55, 3)

    global_orient = torch.from_numpy(global_orient).float()
    body_pose = poses[:, 1:22, :]
    jaw_pose = poses[:, 22:23, :]
    leye_pose = poses[:, 23:24, :]
    reye_pose = poses[:, 24:25, :]
    left_hand_pose = poses[:, 25:40, :]
    right_hand_pose = poses[:, 40:55, :]
    expression = torch.zeros(nframe, 10)

    # create smplx body model
    model_folder = 'body_models/'
    model_type = 'smplx'

    model = smplx.create(model_folder,
                         model_type=model_type,
                         gender=gender,
                         use_face_contour=True,
                         use_pca=False,
                         num_betas=num_betas,
                         num_expression_coeffs=10,
                         ext='npz')

    output = model(transl=transl,
                   betas=betas,
                   global_orient=global_orient,
                   body_pose=body_pose,
                   jaw_pose=jaw_pose,
                   leye_pose=leye_pose,
                   reye_pose=reye_pose,
                   left_hand_pose=left_hand_pose,
                   right_hand_pose=right_hand_pose,
                   expression=expression,
                   return_verts=True)
    vertices = output.vertices.detach().cpu().numpy().squeeze()
    return vertices, model.faces


if __name__ == "__main__":

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--fn1',
                        type=str,
                        help='A mesh file to be checked for self-collisions')
    parser.add_argument('--fn2',
                        type=str,
                        help='A mesh file to be checked for self-collisions')
    parser.add_argument('--ofn',
                        type=str,
                        help='A npz file to save the collision results')
    parser.add_argument('--max_collisions',
                        default=8,
                        type=int,
                        help='The maximum number of bounding box collisions')

    args, _ = parser.parse_known_args()

    max_collisions = args.max_collisions

    vertices1, faces1 = get_smplx_mesh(args.fn1)
    vertices2, faces2 = get_smplx_mesh(args.fn2)

    assert vertices1.shape == vertices2.shape
    assert (faces1 == faces2).all()
    del faces2

    nframe = vertices1.shape[0]
    col_lst = {'frame': [], 'n_col': [], 'r_col': []}
    t0 = time.time()
    for i in tqdm(range(nframe)):
        input_mesh1 = trimesh.Trimesh(vertices1[i, :], faces1)
        input_mesh2 = trimesh.Trimesh(vertices2[i, :], faces1)
        col1, _ = detect_collision(input_mesh1,
                                   max_collisions,
                                   device,
                                   verbose=False)
        col2, _ = detect_collision(input_mesh2,
                                   max_collisions,
                                   device,
                                   verbose=False)

        input_mesh = trimesh.util.concatenate([input_mesh1, input_mesh2])
        col12, input_mesh = detect_collision(input_mesh,
                                             max_collisions,
                                             device,
                                             verbose=False)

        col2 += input_mesh1.faces.shape[0]
        self_collisions = np.concatenate([col1, col2])

        self_col = numpy2set(self_collisions)
        all_col = numpy2set(col12)

        collisions = np.array(list(all_col - self_col))
        n_collisions = len(collisions)
        assert n_collisions == (len(all_col) - len(col1) - len(col2))

        if n_collisions > 0:
            ratio = n_collisions / float(input_mesh.faces.shape[0]) * 100
            col_lst['frame'].append(i)
            col_lst['n_col'].append(n_collisions)
            col_lst['r_col'].append(ratio)
            # print(f'[{i}] Number of collisions = {n_collisions}')
            # print(f'[{i}] Percentage of collisions (%): {ratio}')

    t1 = time.time()
    print(f'Process {nframe} frames: {t1 - t0} s')
    n = len(col_lst['frame'])
    print(f'{n} frames from {nframe} frames have collision between two person')
    avg_ratio = np.mean(col_lst['r_col'])
    print(f'avg collision ratio of {n} frames is: {avg_ratio:.4f}%')
