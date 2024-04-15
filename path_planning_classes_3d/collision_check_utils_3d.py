import numpy as np

def check_collision_line_single_ball(
    line,
    ball_center,
    ball_radius,
    clearance=0.,
):
    """
    - inputs:
        - line: [[x_start, y_start, z_start],[x_end, y_end, z_end]] np.
        - ball_center: [x_center, y_center, z_center] np.
        - ball_radius: scalar > 0.
        - clearance: scalar >= 0
    - outputs:
        - collision: bool.
    - references:
        https://cseweb.ucsd.edu/classes/sp19/cse291-d/Files/CSE291_13_CollisionDetection.pdf
    """
    
    c, r = ball_center, ball_radius+clearance
    p0, p1 = line[0], line[1]
    line = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
    if np.linalg.norm(line)==0:
        return point_in_single_ball(p0, ball_center, ball_radius, clearance)
    d1 = [c[0] - p0[0], c[1] - p0[1], c[2] - p0[2]]
    t = (1 / (line[0] * line[0] + line[1] * line[1] + line[2] * line[2])) * (
                line[0] * d1[0] + line[1] * d1[1] + line[2] * d1[2])
    if t <= 0:
        if (d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]) <= r ** 2: return True
    elif t >= 1:
        d2 = [c[0] - p1[0], c[1] - p1[1], c[2] - p1[2]]
        if (d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]) <= r ** 2: return True
    elif 0 < t < 1:
        x = [p0[0] + t * line[0], p0[1] + t * line[1], p0[2] + t * line[2]]
        k = [c[0] - x[0], c[1] - x[1], c[2] - x[2]]
        if (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) <= r ** 2: return True
    return False


def check_collision_line_single_box(
    line,
    xyzwhd,
    clearance=0.,
):
    """
    - inputs:
        - line: [[x_start, y_start, z_start],[x_end, y_end, z_end]] np.
        - xyzwhd: [x, y, z, w, h, d] np or list.
        - clearance: scalar >= 0
    - outputs:
        - collision: bool.
    - references:
        https://www.gamedeveloper.com/game-platforms/simple-intersection-tests-for-games
    """
    # (p0, p1, dist, aabb)
    # aabb should have the attributes of P, E as center point and extents
    mid = (line[0]+line[1])/2 # mid point
    direction = line[1]-line[0]
    dist = np.linalg.norm(direction)
    if dist == 0:
        return point_in_single_box(line[0], xyzwhd, clearance)
    I = direction / dist  # unit direction
    hl = dist / 2  # radius

    x, y, z, w, h, d = xyzwhd
    aabb_P = [x+w/2, y+h/2, z+d/2]
    aabb_E = [w/2+clearance, h/2+clearance, d/2+clearance]

    T = [aabb_P[0] - mid[0], aabb_P[1] - mid[1], aabb_P[2] - mid[2]]
    # do any of the principal axis form a separting axis?
    if abs(T[0]) > (aabb_E[0] + hl * abs(I[0])): return False
    if abs(T[1]) > (aabb_E[1] + hl * abs(I[1])): return False
    if abs(T[2]) > (aabb_E[2] + hl * abs(I[2])): return False
    # I.cross(s axis) ?
    r = aabb_E[1] * abs(I[2]) + aabb_E[2] * abs(I[1])
    if abs(T[1] * I[2] - T[2] * I[1]) > r: return False
    # I.cross(y axis) ?
    r = aabb_E[0] * abs(I[2]) + aabb_E[2] * abs(I[0])
    if abs(T[2] * I[0] - T[0] * I[2]) > r: return False
    # I.cross(z axis) ?
    r = aabb_E[0] * abs(I[1]) + aabb_E[1] * abs(I[0])
    if abs(T[0] * I[1] - T[1] * I[0]) > r: return False
    return True


def point_in_single_ball(
    point,
    ball_center,
    ball_radius,
    clearance=0.,
):
    """
    - inputs:
        - point: [x, y, z] np.
        - ball_center: [x_center, y_center, z_center] np.
        - ball_radius: scalar > 0.
        - clearance: scalar >= 0
    - outputs:
        - in_ball: bool.
    """
    return np.linalg.norm(point-ball_center)<=ball_radius+clearance


def point_in_single_box(
    point,
    xyzwhd,
    clearance=0.,
):
    """
    - inputs:
        - point: [x, y, z] np or list.
        - xywh: [x, y, z, w, h, d] np or list.
        - clearance: scalar >= 0
    - outputs:
        - in_box: bool.
    """
    x, y, z, w, h, d = xyzwhd
    return x-clearance <= point[0] <= x+w+clearance and \
           y-clearance <= point[1] <= y+h+clearance and \
           z-clearance <= point[2] <= z+d+clearance
            

def check_collision_single_aabb_pair_3d(aabb1, aabb2):
    """
    - inputs:
        - aabb1: [[x1, y1, z1], [x2, y2, z2]] where x2=x1+w, y2=y1+h, z2=z1+d. np or list.
        - aabb2: [[x1, y1, z1], [x2, y2, z2]] where x2=x1+w, y2=y1+h, z2=z1+d. np or list.
    - outputs:
        - collision: bool.
    """
    return (aabb1[0][0] <= aabb2[1][0] and aabb1[1][0] >= aabb2[0][0]) and \
           (aabb1[0][1] <= aabb2[1][1] and aabb1[1][1] >= aabb2[0][1]) and \
           (aabb1[0][2] <= aabb2[1][2] and aabb1[1][2] >= aabb2[0][2])


def check_collsion_aabb_aabbs_3d(aabb, aabbs):
    """
    - inputs:
        - aabb: [[x1, y1, z1], [x2, y2, z2]] where x2=x1+w, y2=y1+h, z2=z1+d. np (2, 3)
        - aabbs: [[x1, y1, z1], [x2, y2, z2]] where x2=x1+w, y2=y1+h, z2=z1+d. np (n_aabbs, 2, 3)
    - outputs:
        - collision: np bool (n_aabbs,).
    """
    collision = (aabb[0,0]<=aabbs[:,1,0])*(aabb[1,0]>=aabbs[:,0,0])*\
                (aabb[0,1]<=aabbs[:,1,1])*(aabb[1,1]>=aabbs[:,0,1])*\
                (aabb[0,2]<=aabbs[:,1,2])*(aabb[1,2]>=aabbs[:,0,2]) # (n_aabbs,)
    return collision


def check_collision_line_balls_boxes(
    line,
    balls,
    boxes,
    clearance=0.,
):
    """
    - inputs:
        - line: [[x_start, y_start, z_start],[x_end, y_end, z_end]] np. (2,3)
        - balls: np (n_balls, 4), (x, y, z, r), or None, which means no obstacles
        - boxes: np (n_boxes, 6), (x, y, z, w, h, d), or None, which means no obstacles
        - clearance: scalar >= 0
    - outputs:
        - collision: bool.
    """
    line_aabb = np.array([np.min(line,axis=0), np.max(line,axis=0)]) # [[x1, y1, z1], [x2, y2, z2]]
    if balls is not None:
        balls_x1 = balls[:,0]-balls[:,3]-clearance
        balls_y1 = balls[:,1]-balls[:,3]-clearance
        balls_z1 = balls[:,2]-balls[:,3]-clearance
        balls_x2 = balls[:,0]+balls[:,3]+clearance
        balls_y2 = balls[:,1]+balls[:,3]+clearance
        balls_z2 = balls[:,2]+balls[:,3]+clearance
        ball_aabbs = np.array([[balls_x1, balls_y1, balls_z1],[balls_x2, balls_y2, balls_z2]]) # (2,3,n_balls)
        ball_aabbs = ball_aabbs.transpose(2,0,1) # (n_balls,2,3)
        ball_aabb_collisions = check_collsion_aabb_aabbs_3d(
            line_aabb,
            ball_aabbs,
        )
        balls_to_check = balls[np.where(ball_aabb_collisions)]
        if len(balls_to_check) > 0:
            for ball_to_check in balls_to_check:
                ball_center, ball_radius = ball_to_check[:3], ball_to_check[3]
                if check_collision_line_single_ball(
                    line,
                    ball_center,
                    ball_radius,
                    clearance,
                ):
                    return True
    if boxes is not None:
        # - boxes: [x, y, z, w, h, d] np. (n_boxes, 6)
        boxes_x1 = boxes[:,0]-clearance
        boxes_y1 = boxes[:,1]-clearance
        boxes_z1 = boxes[:,2]-clearance

        boxes_x2 = boxes[:,0]+boxes[:,3]+clearance
        boxes_y2 = boxes[:,1]+boxes[:,4]+clearance
        boxes_z2 = boxes[:,2]+boxes[:,5]+clearance

        box_aabbs = np.array([[boxes_x1, boxes_y1, boxes_z1],[boxes_x2, boxes_y2, boxes_z2]]) # (2,3,n_boxes)
        box_aabbs = box_aabbs.transpose(2,0,1) # (n_boxes,2,3)
        box_aabb_collisions = check_collsion_aabb_aabbs_3d(
            line_aabb,
            box_aabbs,
        )
        boxes_to_check = boxes[np.where(box_aabb_collisions)]
        if len(boxes_to_check) > 0:
            for box_to_check in boxes_to_check:
                if check_collision_line_single_box(
                    line,
                    box_to_check,
                    clearance,
                ):
                    return True
    return False


def points_in_boxes(
    points,
    boxes,
    clearance=0.,
):
    """
    check whether 3D points are in 3D boxes
    - inputs
        - points: np (n, 3) or tuple (3,)
        - boxes: np (m, 6), (x, y, z, w, h, d), or None, which means no obstacles
        - clearance: scalar >= 0
    - outputs:
        - in_boxes: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp,zp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,3)
    if boxes is None:
        if points_type==tuple:
            in_boxes = False
        else:
            in_boxes = np.zeros(points.shape[0]).astype(bool)
        return in_boxes
    n, m = points.shape[0], boxes.shape[0]
    xp, yp, zp = points[:,0:1], points[:,1:2], points[:,2:3] # (n, 1)
    xp, yp, zp = xp*np.ones((1,m)), yp*np.ones((1,m)), zp*np.ones((1,m)) # (n,m)
    xmin, ymin, zmin, w, h, d = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3], boxes[:,4], boxes[:,5] # (m,)
    xmax, ymax, zmax = xmin+w+clearance, ymin+h+clearance, zmin+d+clearance
    xmin, ymin, zmin = xmin-clearance, ymin-clearance, zmin-clearance
    xmin, ymin, zmin, xmax, ymax, zmax = \
        xmin*np.ones((n,1)), ymin*np.ones((n,1)), zmin*np.ones((n,1)), \
        xmax*np.ones((n,1)), ymax*np.ones((n,1)), zmax*np.ones((n,1)) # (n,m)
    in_boxes = (xmin<=xp)*(xp<=xmax)*(ymin<=yp)*(yp<=ymax)*(zmin<=zp)*(zp<=zmax) # (n,m)
    in_boxes = np.sum(in_boxes, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_boxes[0]
    return in_boxes

def points_in_balls(
    points,
    balls,
    clearance=0.,
):
    """
    check whether 3D points are in 3D balls
    - inputs
        - points: np (n, 3) or tuple (3,)
        - balls: np (m, 4), (x, y, z, r), or None, which means no obstacles
        - clearance: scalar >= 0
    - outputs:
        - in_balls: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp,zp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,3)
    if balls is None:
        if points_type==tuple:
            in_balls = False
        else:
            in_balls = np.zeros(points.shape[0]).astype(bool)
        return in_balls
    n, m = points.shape[0], balls.shape[0]
    xp, yp, zp = points[:,0:1], points[:,1:2], points[:,2:3] # (n, 1)
    xp, yp, zp = xp*np.ones((1,m)), yp*np.ones((1,m)), zp*np.ones((1,m)) # (n,m)

    xc, yc, zc, rc = balls[:,0], balls[:,1], balls[:,2], balls[:,3] # (m,)
    rc = rc + clearance

    in_balls = (xp-xc)**2+(yp-yc)**2+(zp-zc)**2<rc**2 # (n,m)
    in_balls = np.sum(in_balls, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_balls[0]
    return in_balls


def points_in_balls_boxes(
    points,
    balls,
    boxes,
    clearance=0.,
):
    """
    check whether 3D points are in 3D balls or 3D boxes
    - inputs:
        - points: np (n, 3) or tuple (3,)
        - balls: np (m, 4), (x, y, z, r), or None, which means no obstacles
        - boxes: np (m, 6), (x, y, z, w, h, d), or None, which means no obstacles
        - clearance: scalar >= 0
    - outputs:
        - in_balls_or_boxes: np bool (n, ) or bool value
    """
    in_balls = points_in_balls(
        points,
        balls,
        clearance,
    )
    in_boxes = points_in_boxes(
        points,
        boxes,
        clearance,
    )
    if type(points)==tuple:
        return bool(in_balls+in_boxes)
    else:
        return (in_balls+in_boxes).astype(bool)

def points_in_range_3d(
    points,
    x_range,
    y_range,
    z_range,
    clearance=0.,
):
    """
    - inputs:
        - points: np (n, 3) or tuple (3,)
        - x_range: (xmin, xmax)
        - y_range: (ymin, ymax)
        - z_range: (zmin, zmax)
        - clearance: scalar >= 0, clearance for boundary.
    - outputs:
        - np bool (n, ) or bool value
    """
    range_xyzwhd = np.array([[x_range[0], y_range[0], z_range[0], x_range[1]-x_range[0], y_range[1]-y_range[0], z_range[1]-z_range[0]]]) # (1,6)
    # shrink the range for boundary clearance
    return points_in_boxes(
        points,
        range_xyzwhd,
        clearance=-clearance,
    )
        
def points_validity_3d(
    points,
    ball_obstacles,
    box_obstacles,
    x_range,
    y_range,
    z_range,
    obstacle_clearance=0.,
    range_clearance=0.,
):
    """
    - inputs:
        - points: np (n, 3) or tuple (3,)
        - ball_obstacles: np (m, 4), (x, y, z, r), or None, which means no obstacles
        - box_obstacles: np (m, 6), (x, y, z, w, h, d), or None, which means no obstacles
        - x_range: (xmin, xmax)
        - y_range: (ymin, ymax)
        - z_range: (zmin, zmax)
        - obstacle_clearance: scalar >= 0, clearance for obstacles.
        - range_clearance: scalar >= 0, clearance for range.
    - outputs:
        - np bool (n, ) or bool value
    """
    in_range = points_in_range_3d(
        points,
        x_range,
        y_range,
        z_range,
        clearance=range_clearance,
    )
    in_balls = points_in_balls(
        points,
        ball_obstacles,
        clearance=obstacle_clearance,
    )
    in_boxes = points_in_boxes(
        points,
        box_obstacles,
        clearance=obstacle_clearance,
    )
    # in range, and not in_balls, and not in_boxes
    if type(points)==tuple:
        return bool(in_range*(1-in_balls)*(1-in_boxes))
    else:
        return (in_range*(1-in_balls)*(1-in_boxes)).astype(bool)