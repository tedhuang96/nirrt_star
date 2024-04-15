from collections import deque

import numpy as np

def get_boundary_mask(
    pc,
    path_mask,
    unvisited_mask,
    boundary_distance_threshold,
):
    """
    - inputs:
        - pc: np float32 (n_points, 2) for XY or (n_points, 3) for XYZ
        - path_mask: (n_points, ) np float32 0., 1.
        - unvisited_mask: (n_points, ) np float32. 0., 1. unvisited_mask and path_mask may not complement each other.
        - boundary_distance_threshold: float scalar. If any unvisited point is with in this distance of a path point, that path point is a boundary point.
    - outputs:
        - boundary_mask: (n_points, ) np float32 0., 1.
    """
    path_points = pc[path_mask.astype(bool)] # (n_path_points, C)
    unvisited_points = pc[unvisited_mask.astype(bool)] # (n_unvisited_points, C)
    dist_mat = path_points[:,np.newaxis]-unvisited_points # (n_path_points, n_unvisited_points, C)
    dist_mat = np.linalg.norm(dist_mat, axis=2) # (n_path_points, n_unvisited_points)
    boundary_mask_on_path_points = (dist_mat < boundary_distance_threshold).astype(float).sum(axis=1)>0 # (n_path_points,)
    bounday_point_indices = np.where(path_mask.astype(bool))[0][boundary_mask_on_path_points]
    boundary_mask = np.zeros(len(pc))
    boundary_mask[bounday_point_indices] = 1
    boundary_mask = boundary_mask.astype(np.float32)
    return boundary_mask

def bfs_point_cloud(
    pc,
    path_mask,
    x_start,
    x_goal,
    step_len,
):
    """
    - inputs:
        - pc: np float32 (n_points, C). (n_points, 2) for XY or (n_points, 3) for XYZ
        - path_mask: (n_points, ) np float32 0., 1.
        - x_start: np float32 (C,).
        - x_goal: np float32 (C,).
        - step_len: float scalar.
    - outputs:
        - has_path: bool
        - visited_mask: (n_points, ) np float32 0., 1.
    """
    path_points = pc[path_mask.astype(bool)] # (n_path_points, C)
    vertices = np.concatenate([x_start[np.newaxis], x_goal[np.newaxis], path_points], axis=0) # (2+n, c) # start 0, goal 1
    dist_mat = vertices[:,np.newaxis]-vertices # (2+n, 2+n, C)
    adj_mat = np.linalg.norm(dist_mat, axis=2)<step_len # (2+n, 2+n)
    visited = set()
    queue = deque([0])
    visited.add(0)
    while queue:
        vertex = queue.popleft()
        neighbors = np.where(adj_mat[vertex])[0]
        for neighbor in neighbors:
            if neighbor == 1:
                has_path = True
                visited = np.array(list(visited)[1:]).astype(int)-2 # path point indices 2->0. Remove start
                visited_point_indices = np.where(path_mask.astype(bool))[0][visited]
                visited_mask = np.zeros(len(pc))
                visited_mask[visited_point_indices] = 1
                visited_mask = visited_mask.astype(np.float32)
                return has_path, visited_mask
            if neighbor not in visited:
                queue.append(neighbor)   # Add unvisited neighbors to the queue
                visited.add(neighbor)    # Mark the neighbor as visited
    has_path = False
    visited = np.array(list(visited)[1:]).astype(int)-2 # path point indices 2->0. Remove start
    visited_point_indices = np.where(path_mask.astype(bool))[0][visited]
    visited_mask = np.zeros(len(pc))
    visited_mask[visited_point_indices] = 1
    visited_mask = visited_mask.astype(np.float32)
    return has_path, visited_mask


def bfs_point_cloud_visualization(
    pc,
    path_mask,
    x_start,
    x_goal,
    step_len,
):
    """
    - inputs:
        - pc: np float32 (n_points, C). (n_points, 2) for XY or (n_points, 3) for XYZ
        - path_mask: (n_points, ) np float32 0., 1.
        - x_start: np float32 (C,).
        - x_goal: np float32 (C,).
        - step_len: float scalar.
    - outputs:
        - has_path: bool
        - path_line: None or (n_path, C) from start to goal through path points. np float32.
        - visited_mask: (n_points, ) np float32 0., 1.
    """
    path_points = pc[path_mask.astype(bool)] # (n_path_points, C)
    vertices = np.concatenate([x_start[np.newaxis], x_goal[np.newaxis], path_points], axis=0) # (2+n, c) # start 0, goal 1
    dist_mat = vertices[:,np.newaxis]-vertices # (2+n, 2+n, C)
    adj_mat = np.linalg.norm(dist_mat, axis=2)<step_len # (2+n, 2+n)
    visited = set()
    queue = deque([0])
    visited.add(0)
    parents = {}

    while queue:
        vertex = queue.popleft()
        neighbors = np.where(adj_mat[vertex])[0]
        for neighbor in neighbors:
            if neighbor == 1:
                has_path = True
                path_vertex = vertex
                path_line = [1, path_vertex]
                while path_vertex != 0:
                    parent_vertex = parents[path_vertex]
                    path_vertex = parent_vertex
                    path_line.append(path_vertex)
                path_line.reverse() # [0, ..., 1] # [start, ..., goal]
                path_line = vertices[np.array(path_line)]
                visited = np.array(list(visited)[1:]).astype(int)-2 # path point indices 2->0. Remove start
                visited_point_indices = np.where(path_mask.astype(bool))[0][visited]
                visited_mask = np.zeros(len(pc))
                visited_mask[visited_point_indices] = 1
                visited_mask = visited_mask.astype(np.float32)
                return has_path, path_line, visited_mask
            if neighbor not in visited:
                queue.append(neighbor)   # Add unvisited neighbors to the queue
                visited.add(neighbor)    # Mark the neighbor as visited
                parents[neighbor] = vertex
    has_path = False
    path_line = None
    visited = np.array(list(visited)[1:]).astype(int)-2 # path point indices 2->0. Remove start
    visited_point_indices = np.where(path_mask.astype(bool))[0][visited]
    visited_mask = np.zeros(len(pc))
    visited_mask[visited_point_indices] = 1
    visited_mask = visited_mask.astype(np.float32)
    return has_path, path_line, visited_mask


def select_heuristic_boundary_point(
    pc,
    boundary_mask,
    x_start,
    x_goal,
    cost_from_start_rank_weight=1,
):
    """
    Select an edge point based on heuristic with minimum total cost and maximum g (furthest from start point).
    - inputs:
        - pc: np float32 (n_points, C).
        - boundary_mask: np float32 (n_points,) 0., 1.
        - x_start: np float32 (C,)
        - x_goal: np float32 (C,)
    - outputs:
        - boundary_point_index: index in pc for the selected boundary point.
        - boundary_point: np float32 (C,) the selected boundary point.
        - boundary_point_heuristic: list (n_bound_points,) heuristic values for all boundary points.
    """
    boundary_points = pc[boundary_mask.astype(bool)]
    if len(boundary_points)==0:
        # No boundary points
        return None, None, None
    
    heuristic_cost_from_start = np.linalg.norm(boundary_points-x_start, axis=1) # (n_boundary_points,)
    heuristic_cost_to_goal = np.linalg.norm(boundary_points-x_goal, axis=1) # (n_boundary_points,)

    total_cost_rank = np.argsort(heuristic_cost_from_start+heuristic_cost_to_goal)
    cost_from_start_rank = np.flip(np.argsort(heuristic_cost_from_start))

    total_cost_rank_dict = {total_cost_rank[i]: i for i in range(len(total_cost_rank))}
    cost_from_start_rank_dict = {cost_from_start_rank[i]: i for i in range(len(cost_from_start_rank))}
    
    boundary_point_heuristic = [-(total_cost_rank_dict[edge_i]+cost_from_start_rank_weight*cost_from_start_rank_dict[edge_i]) for edge_i in range(len(heuristic_cost_to_goal))]


    boundary_point_index = np.where(boundary_mask)[0][np.argmax(boundary_point_heuristic)]
    boundary_point = pc[boundary_point_index]

    return boundary_point_index, boundary_point, boundary_point_heuristic
