import numpy as np
import open3d as o3d

from path_planning_utils_3d.collision_check_utils import points_in_AABB_3d, points_in_ball_3d
from path_planning_classes_3d.collision_check_utils_3d import points_in_balls_boxes, points_validity_3d


def farthest_point_sample(points, npoint):
    """
    Slower than open3d version.
    Input:
        points: pointcloud data, [B, N, C] or [N, C]
        npoint: number of samples
    Return:
        downsampled_points: [B, npoint, C] or [npoint, C]
        downsampled_indices: sampled pointcloud index, [B, npoint]  or [npoint,]
    """
    points_shape = len(points.shape)
    if points_shape == 2:
        points = points[np.newaxis,:]
    B, N, C = points.shape
    centroids = np.zeros((B, npoint)).astype(int)
    distance = np.ones((B, N)) * 1e10
    farthest = np.random.randint(0, N, (B,))
    batch_indices = np.arange(B)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].reshape(B, 1, C)
        dist = np.sum((points - centroid) ** 2, -1) # euclidean distance
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance,-1)
    downsampled_indices = centroids
    downsampled_points_batch_indices = batch_indices[:,np.newaxis]*np.ones((1, npoint)).astype(int)
    downsampled_points = points[downsampled_points_batch_indices, downsampled_indices]
    if points_shape == 2:
        downsampled_points = downsampled_points[0]
        downsampled_indices = downsampled_indices[0]
    return downsampled_points, downsampled_indices

def farthest_point_sample_open3d(points, npoint):
    """
    Input:
        points: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        downsampled_points: [npoint, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.farthest_point_down_sample(num_samples=npoint)
    downsampled_points = np.asarray(pcd.points)
    return downsampled_points

def generate_rectangle_point_cloud_3d_v1(
    env,
    n_points,
    over_sample_scale=5,
    use_open3d=True,
):
    """
    When creating point cloud, no clearance. Consistent with 2D. We want more points around optimal path to be selected. We may erase them later.
    But we want to keep the topology. If we need to remove topology, we can use clearance, but that is saved for later work.
    - outputs:
        - point_cloud: (n_points, 2)
    """
    point_cloud = np.random.uniform(
        low=env.boundary_no_clearance[:3],
        high=env.boundary_no_clearance[3:],
        size=(n_points*over_sample_scale, 3),
    )
    point_cloud_in_aabb = points_in_AABB_3d(point_cloud, env.box_obstacles, clearance=0)
    point_cloud_in_ball = points_in_ball_3d(point_cloud, env.balls_no_clearance, clearance=0)
    point_cloud = point_cloud[np.where((1-point_cloud_in_aabb)*(1-point_cloud_in_ball))[0]]
    if use_open3d:
        point_cloud = farthest_point_sample_open3d(point_cloud, n_points)
    else:
        point_cloud, _ = farthest_point_sample(point_cloud, n_points)
    return point_cloud



def generate_rectangle_point_cloud_3d(
    env,
    n_points,
    over_sample_scale=5,
    use_open3d=True,
    clearance=0,
):
    """
    When creating point cloud, no clearance. Consistent with 2D. We want more points around optimal path to be selected. We may erase them later.
    But we want to keep the topology. If we need to remove topology, we can use clearance, but that is saved for later work.
    - outputs:
        - point_cloud: (n_points, 2)
    """
    point_cloud = np.random.uniform(
        low=(env.x_range[0]+clearance, env.y_range[0]+clearance, env.z_range[0]+clearance),
        high=(env.x_range[1]-clearance, env.y_range[1]-clearance, env.z_range[1]-clearance),
        size=(n_points*over_sample_scale, 3),
    )
    in_obs = points_in_balls_boxes(
        point_cloud,
        np.array(env.obs_ball).astype(np.float64),
        np.array(env.obs_box).astype(np.float64),
        clearance=clearance, # * can be adjusted
    )
    point_cloud = point_cloud[(1-in_obs).astype(bool)]
    if len(point_cloud) > n_points:
        if use_open3d:
            point_cloud = farthest_point_sample_open3d(point_cloud, n_points)
        else:
            point_cloud, _ = farthest_point_sample(point_cloud, n_points)
    return point_cloud



def RotationToWorldFrame(x_start, x_goal, L):
    # S0(n): such that the x_start and x_goal are the center points
    '''
    - inputs:
        - x_start: np (3,)
        - x_goal: np (3,)
        - L: distance between x_start and x_goal
    '''
    a1 = (x_goal - x_start) / L
    M = np.outer(a1,[1,0,0])
    U, S, V = np.linalg.svd(M)
    C = U@np.diag([1, 1, np.linalg.det(U)*np.linalg.det(V)])@V.T
    return C


def ellipsoid_point_cloud_sampling_3d(
    start_point,
    goal_point,
    max_min_ratio,
    env,
    n_points=1000,
    n_raw_samples=10000,
    clearance=0,
):
    """
    - inputs
        - start_point: np (3,)
        - goal_point: np (3,)
        - max_min_ratio: scalar >= 1.0
        - env
    - outputs
        - point_cloud: np (n_points, 3)
    """
    c_min = np.linalg.norm(goal_point-start_point)
    C = RotationToWorldFrame(start_point, goal_point, c_min)
    x_center = (start_point+goal_point)/2.  # (3,)
    c_max = c_min*max_min_ratio
    if c_max ** 2 - c_min ** 2<0:
        eps = 1e-6
    else:
        eps = 0
    r = np.zeros(3)
    r[0] = c_max /2
    for i in [1,2]:
        r[i] = np.sqrt(c_max**2-c_min**2+eps) / 2
    L = np.diag(r) # R3*3

    radius = np.random.uniform(0.0, 1.0, n_raw_samples)
    theta = np.random.uniform(0, np.pi, n_raw_samples)
    phi = np.random.uniform(0, 2 * np.pi, n_raw_samples)
    samples_x = radius * np.sin(theta) * np.cos(phi)
    samples_y = radius * np.sin(theta) * np.sin(phi)
    samples_z = radius * np.cos(theta)
    samples = np.array([samples_x, samples_y, samples_z]).T # (n, 3)
    x_rand = np.dot(np.dot(C, L), samples.T).T + x_center # (n,3)
    point_cloud = x_rand

    if len(env.obs_ball)>0:
        obs_ball = np.array(env.obs_ball).astype(np.float64)
    else:
        obs_ball = None
    if len(env.obs_box)>0:
        obs_box = np.array(env.obs_box).astype(np.float64)
    else:
        obs_box = None
    valid_flag = points_validity_3d(
        point_cloud,
        obs_ball,
        obs_box,
        env.x_range,
        env.y_range,
        env.z_range,
        obstacle_clearance=clearance,
        range_clearance=clearance,
    )

    point_cloud = point_cloud[valid_flag]
    if len(point_cloud) > n_points:
        # downsample
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd = pcd.farthest_point_down_sample(num_samples=n_points)
        point_cloud = np.asarray(pcd.points)
    return point_cloud