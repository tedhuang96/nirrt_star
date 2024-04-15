import math

import numpy as np
import open3d as o3d

from path_planning_classes.collision_check_utils import points_in_range

def get_binary_mask(env_img):
    """
    - inputs:
        - env_img: np (img_height, img_width, 3)
        - binary_mask: np float 0. or 1. (img_height, img_width)
    """
    env_dims = env_img.shape[:2]
    binary_mask = np.zeros(env_dims).astype(float)
    binary_mask[env_img[:,:,0]!=0]=1
    return binary_mask


def get_point_cloud_mask_around_points(
    point_cloud,
    points,
    neighbor_radius=3,
    ):
    # point_cloud (n, C)
    # points (m, C), m can be 1
    dist = point_cloud[:,np.newaxis] - points # (n,m,C)
    dist = np.linalg.norm(dist,axis=2) # (n,m) # euclidean distance
    neighbor_mask = dist<neighbor_radius # (n, m)
    around_points_mask = np.sum(neighbor_mask,axis=1)>0 # (n,)
    return around_points_mask


# *** Rectangular sampling ***
def generate_rectangle_point_cloud(
    binary_mask,
    n_points,
    over_sample_scale=5,
):
    """
    - outputs:
        - point_cloud: (n_points, 2)
    """
    img_height, img_width = binary_mask.shape
    # oversampling in whole region
    point_cloud = np.random.uniform(
        low=[0, 0],
        high=[img_width, img_height],
        size=(n_points*over_sample_scale, 2),
    )
    # remove points in occupied space
    point_cloud_pix = point_cloud.astype(int)
    point_cloud_neighbors = []
    for i in range(0, 2):
        for j in range(0, 2):
            point_cloud_neighbors.append(point_cloud_pix+np.array([i,j]))
    pix_nei_offset = np.array([[0,0],[0,1],[1,0],[1,1]])[:,np.newaxis] # (4,1,2)
    point_cloud_pix_nei = point_cloud_pix + pix_nei_offset # (4, n, 2)
    point_cloud_pix_nei = point_cloud_pix_nei.reshape(-1,2)
    point_cloud_pix_nei[:,1] = np.clip(point_cloud_pix_nei[:,1], 0, img_height-1)
    point_cloud_pix_nei[:,0] = np.clip(point_cloud_pix_nei[:,0], 0, img_width-1)
    point_cloud_occupied_mask = np.prod(
        binary_mask[point_cloud_pix_nei[:,1],point_cloud_pix_nei[:,0]].reshape(4,-1),
        axis=0,
    ) # (n,)
    point_cloud = point_cloud[point_cloud_occupied_mask.nonzero()[0]]
    # downsample
    point_cloud_fake_z = np.concatenate([point_cloud, np.zeros((point_cloud.shape[0],1))],axis=1) # (n,3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_fake_z)
    pcd = pcd.farthest_point_down_sample(num_samples=n_points)
    point_cloud = np.asarray(pcd.points)[:,:2]
    return point_cloud


# *** Ellipse sampling ***
def RotationToWorldFrame(start_point, goal_point, L):
    """
    - inputs:
        - start_point: np float64 (2,)
        - goal_point: np float64 (2,)
        - L: scalar
    - outputs:
        - C: rotation matrix, np float64 (3,3)
    """
    a1 = (goal_point - start_point)/L
    a1 = np.concatenate([a1, np.array([0.])], axis=0)[:,np.newaxis] # (3,1)
    e1 = np.array([[1.0], [0.0], [0.0]])
    M = a1 @ e1.T
    U, _, V_T = np.linalg.svd(M, True, True)
    C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T
    return C

def get_distance_and_angle(start_point, goal_point):
    """
    - inputs:
        - start_point: np float64 (2,)
        - goal_point: np float64 (2,)
    """
    dx, dy = goal_point - start_point
    return math.hypot(dx, dy), math.atan2(dy, dx)


def ellipsoid_point_cloud_sampling(
    start_point,
    goal_point,
    max_min_ratio,
    binary_mask,
    n_points=1000,
    n_raw_samples=10000,
):
    """
    - inputs
        - start_point: np (2,)
        - goal_point: np (2,)
        - max_min_ratio: scalar >= 1.0
        - binary_mask: 0-1 mask (img_height, img_width)
    - outputs
        - point_cloud: np (n_points, 2)
    """
    c_min, theta = get_distance_and_angle(start_point, goal_point)
    C = RotationToWorldFrame(start_point, goal_point, c_min)
    x_center = (start_point+goal_point)/2.
    x_center = np.concatenate([x_center, np.array([0.])], axis=0) # (3,)
    c_max = c_min*max_min_ratio
    if c_max ** 2 - c_min ** 2<0:
        eps = 1e-6
    else:
        eps = 0
    r = [c_max / 2.0,
        math.sqrt(c_max ** 2 - c_min ** 2+eps) / 2.0,
        math.sqrt(c_max ** 2 - c_min ** 2+eps) / 2.0]
    L = np.diag(r)

    samples = np.random.uniform(-1, 1, size=(n_raw_samples, 2))
    samples = samples[np.linalg.norm(samples, axis=1) <= 1]
    samples = np.concatenate([samples, np.zeros((len(samples),1))], axis=1) # (n, 3)

    x_rand = np.dot(np.dot(C, L), samples.T).T + x_center
    point_cloud = x_rand[:,:2]
    # remove points in occupied space
    point_cloud_pix = point_cloud.astype(int)
    point_cloud_neighbors = []
    for i in range(0, 2):
        for j in range(0, 2):
            point_cloud_neighbors.append(point_cloud_pix+np.array([i,j]))
    pix_nei_offset = np.array([[0,0],[0,1],[1,0],[1,1]])[:,np.newaxis] # (4,1,2)
    point_cloud_pix_nei = point_cloud_pix + pix_nei_offset # (4, n, 2)
    point_cloud_pix_nei = point_cloud_pix_nei.reshape(-1,2)
    img_height, img_width = binary_mask.shape
    point_cloud_pix_nei[:,1] = np.clip(point_cloud_pix_nei[:,1], 0, img_height-1)
    point_cloud_pix_nei[:,0] = np.clip(point_cloud_pix_nei[:,0], 0, img_width-1)
    point_cloud_occupied_mask = np.prod(
        binary_mask[point_cloud_pix_nei[:,1],point_cloud_pix_nei[:,0]].reshape(4,-1),
        axis=0,
    ) # (n,)
    point_cloud = point_cloud[point_cloud_occupied_mask.nonzero()[0]]
    x_range = (0, img_width)
    y_range = (0, img_height)
    point_cloud_in_range_mask = points_in_range(
        point_cloud,
        x_range,
        y_range,
        clearance=0,
    )
    point_cloud = point_cloud[point_cloud_in_range_mask]
    if len(point_cloud) > n_points:
        # downsample
        point_cloud_fake_z = np.concatenate([point_cloud, np.zeros((point_cloud.shape[0],1))],axis=1) # (n,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_fake_z)
        pcd = pcd.farthest_point_down_sample(num_samples=n_points)
        point_cloud = np.asarray(pcd.points)[:,:2]
    return point_cloud