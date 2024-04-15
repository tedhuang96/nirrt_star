import numpy as np


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def line_intersection(line1, line2):
    """
    - inputs
        - line1: [[x_start, y_start],[x_end,y_end]] np or list.
        - line2: [[x_start, y_start],[x_end,y_end]] np or list.
    - outputs
        - intersection: bool.
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = det(xdiff, ydiff)
    if div == 0:
        return False
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    eps = 1e-6 # * fixed bug e.g. line2[0][1] = line2[1][1] = 15, y = 14.999999999999998 and return False
    if min(line1[0][0], line1[1][0])-eps <= x <= max(line1[0][0], line1[1][0])+eps and \
            min(line1[0][1], line1[1][1])-eps <= y <= max(line1[0][1], line1[1][1])+eps and \
            min(line2[0][0], line2[1][0])-eps <= x <= max(line2[0][0], line2[1][0])+eps and \
            min(line2[0][1], line2[1][1])-eps <= y <= max(line2[0][1], line2[1][1])+eps:
        return True
    return False


def check_collision_line_single_circle(
    line,
    circle_center,
    circle_radius,
    clearance=0,
):
    """
    - inputs:
        - line: [[x_start, y_start],[x_end,y_end]] np.
        - circle_center: [x_center, y_center] np.
        - circle_radius: scalar > 0.
        - clearance: scalar >= 0
    - outputs:
        - collision: bool.
    """
    circle_radius_with_clearance = circle_radius+clearance
    line_vector = line[1] - line[0]
    line_length = np.linalg.norm(line_vector)
    if line_length == 0:
        return point_in_single_circle(line[0], circle_center, circle_radius, clearance)
    line_direction = line_vector / line_length
    start_to_center = circle_center - line[0]
    projection = np.dot(start_to_center, line_direction)
    closest_point = np.clip(projection, 0, line_length) * line_direction + line[0]
    distance = np.linalg.norm(np.array(circle_center) - closest_point)
    if distance <= circle_radius_with_clearance:
        return True
    return False


def point_in_single_circle(
    point,
    circle_center,
    circle_radius,
    clearance=0,
):
    """
    - inputs:
        - point: [x, y] np.
        - circle_center: [x_center, y_center] np.
        - circle_radius: scalar > 0.
        - clearance: scalar >= 0
    - outputs:
        - in_circle: bool.
    """
    return np.linalg.norm(point-circle_center)<=circle_radius+clearance


def point_in_single_rectangle(
    point,
    xywh,
    clearance=0,
):
    """
    - inputs:
        - point: [x, y] np or list.
        - xywh: [x, y, w, h] np or list.
        - clearance: scalar >= 0
    - outputs:
        - in_rectangle: bool.
    """
    x, y, w, h = xywh
    return x-clearance <= point[0] <= x+w+clearance and y-clearance <= point[1] <= y+h+clearance
            

def check_collision_line_single_rectangle(
    line,
    xywh,
    clearance=0,
):
    """
    - inputs:
        - line: [[x_start, y_start],[x_end,y_end]] np.
        - xywh: [x, y, w, h] np or list.
        - clearance: scalar >= 0
    - outputs:
        - collision: bool.
    """
    if point_in_single_rectangle(line[0], xywh, clearance) or \
        point_in_single_rectangle(line[1], xywh, clearance):
        return True
    x, y, w, h = xywh
    rect_points = np.array([
        [x-clearance, y-clearance],
        [x+w+clearance, y-clearance],
        [x+w+clearance, y+h+clearance],
        [x-clearance, y+h+clearance],
    ])
    rect_lines = np.array([
        [rect_points[0], rect_points[1]],
        [rect_points[1], rect_points[2]],
        [rect_points[2], rect_points[3]],
        [rect_points[3], rect_points[0]],
    ])
    for rect_line in rect_lines:
        if line_intersection(line, rect_line):
            return True
    return False


def check_collision_single_aabb_pair(aabb1, aabb2):
    """
    - inputs:
        - aabb1: [[x1, y1], [x2, y2]] where x2=x1+w, y2=y1+h. np or list.
        - aabb2: [[x1, y1], [x2, y2]] where x2=x1+w, y2=y1+h. np or list.
    - outputs:
        - collision: bool.
    """
    return (aabb1[0][0] <= aabb2[1][0] and aabb1[1][0] >= aabb2[0][0]) and \
       (aabb1[0][1] <= aabb2[1][1] and aabb1[1][1] >= aabb2[0][1])


def check_collsion_aabb_aabbs(aabb, aabbs):
    """
    - inputs:
        - aabb: [[x1, y1], [x2, y2]] where x2=x1+w, y2=y1+h. np (2, 2)
        - aabbs: [[x1, y1], [x2, y2]] where x2=x1+w, y2=y1+h. np (n_aabbs, 2, 2)
    - outputs:
        - collision: np bool (n_aabbs,).
    """
    collision = (aabb[0,0]<=aabbs[:,1,0])*(aabb[1,0]>=aabbs[:,0,0])*\
        (aabb[0,1]<=aabbs[:,1,1])*(aabb[1,1]>=aabbs[:,0,1]) # (n_aabbs,)
    return collision


def check_collision_line_circles_rectangles(
    line,
    circles,
    rectangles,
    clearance=0,
):
    """
    - inputs:
        - line: [[x_start, y_start],[x_end,y_end]] np. (2,2)
        - circles: [x_center, y_center, radius] np. (n_circles, 3) or None
        - rectangles: [x, y, w, h] np. (n_rectangles, 4) or None
        - clearance: scalar >= 0
    - outputs:
        - collision: bool.
    """
    line_aabb = np.array([[min(line[0,0], line[1,0]), min(line[0,1], line[1,1])],
                          [max(line[0,0], line[1,0]), max(line[0,1], line[1,1])]]) # [[x1, y1], [x2, y2]]
    if circles is not None:
        circles_x1 = circles[:,0]-circles[:,2]-clearance
        circles_y1 = circles[:,1]-circles[:,2]-clearance
        circles_x2 = circles[:,0]+circles[:,2]+clearance
        circles_y2 = circles[:,1]+circles[:,2]+clearance
        circle_aabbs = np.array([[circles_x1, circles_y1],[circles_x2, circles_y2]]) # (2,2,n_circles)
        circle_aabbs = circle_aabbs.transpose(2,0,1) # (n_circles,2,2)
        circle_aabb_collisions = check_collsion_aabb_aabbs(
            line_aabb,
            circle_aabbs,
        )
        circles_to_check = circles[np.where(circle_aabb_collisions)]
        if len(circles_to_check) > 0:
            for circle_to_check in circles_to_check:
                circle_center, circle_radius = circle_to_check[:2], circle_to_check[2]
                if check_collision_line_single_circle(
                    line,
                    circle_center,
                    circle_radius,
                    clearance,
                ):
                    return True
    if rectangles is not None:
        # - rectangles: [x, y, w, h] np. (n_rectangles, 4)
        rectangles_x1 = rectangles[:,0]-clearance
        rectangles_y1 = rectangles[:,1]-clearance
        rectangles_x2 = rectangles[:,0]+rectangles[:,2]+clearance
        rectangles_y2 = rectangles[:,1]+rectangles[:,3]+clearance
        rectangle_aabbs = np.array([[rectangles_x1, rectangles_y1],[rectangles_x2, rectangles_y2]]) # (2,2,n_rectangles)
        rectangle_aabbs = rectangle_aabbs.transpose(2,0,1) # (n_rectangles,2,2)
        rectangle_aabb_collisions = check_collsion_aabb_aabbs(
            line_aabb,
            rectangle_aabbs,
        )
        rectangles_to_check = rectangles[np.where(rectangle_aabb_collisions)]
        if len(rectangles_to_check) > 0:
            for rectangle_to_check in rectangles_to_check:
                if check_collision_line_single_rectangle(
                    line,
                    rectangle_to_check,
                    clearance,
                ):
                    return True
    return False


def points_in_rectangles(
    points,
    rectangles,
    clearance=0,
):
    """
    check whether 2D points are in 2D rectangles
    - inputs
        - points: np (n, 2) or tuple (2,)
        - rectangles: np (m, 4), (x, y, w, h), or None, which means no obstacles
        - clearance: scalar >= 0
    - outputs:
        - in_rectangles: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,2)
    if rectangles is None:
        if points_type==tuple:
            in_rectangles = False
        else:
            in_rectangles = np.zeros(points.shape[0]).astype(bool)
        return in_rectangles
    n, m = points.shape[0], rectangles.shape[0]
    xp, yp = points[:,0:1], points[:,1:2] # (n, 1)
    xp, yp = xp*np.ones((1,m)), yp*np.ones((1,m)) # (n,m)
    xmin, ymin, w, h = rectangles[:,0], rectangles[:,1], rectangles[:,2], rectangles[:,3] # (m,)
    xmax, ymax = xmin+w+clearance, ymin+h+clearance
    xmin, ymin = xmin-clearance, ymin-clearance
    xmin, ymin, xmax, ymax = \
        xmin*np.ones((n,1)), ymin*np.ones((n,1)), xmax*np.ones((n,1)), ymax*np.ones((n,1)) # (n,m)
    in_rectangles = (xmin<=xp)*(xp<=xmax)*(ymin<=yp)*(yp<=ymax) # (n,m)
    in_rectangles = np.sum(in_rectangles, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_rectangles[0]
    return in_rectangles

def points_in_circles(
    points,
    circles,
    clearance=0,
):
    """
    check whether 2D points are in 2D circles
    - inputs
        - points: np (n, 2) or tuple (2,)
        - circles: np (m, 3), (x, y, r), or None, which means no obstacles
        - clearance: scalar >= 0
    - outputs:
        - in_circles: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,2)
    if circles is None:
        if points_type==tuple:
            in_circles = False
        else:
            in_circles = np.zeros(points.shape[0]).astype(bool)
        return in_circles
    n, m = points.shape[0], circles.shape[0]
    xp, yp = points[:,0:1], points[:,1:2] # (n, 1)
    xp, yp = xp*np.ones((1,m)), yp*np.ones((1,m)) # (n,m)

    xc, yc, rc = circles[:,0], circles[:,1], circles[:,2] # (m,)
    rc = rc + clearance

    in_circles = (xp-xc)**2+(yp-yc)**2<rc**2 # (n,m)
    in_circles = np.sum(in_circles, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_circles[0]
    return in_circles


def points_in_circles_rectangles(
    points,
    circles,
    rectangles,
    clearance=0,
):
    """
    check whether 2D points are in 2D circles or 2D rectangles
    - inputs:
        - points: np (n, 2) or tuple (2,)
        - circles: np (m, 3), (x, y, r), or None, which means no obstacles
        - rectangles: np (m, 4), (x, y, w, h), or None, which means no obstacles
        - clearance: scalar >= 0
    - outputs:
        - in_circles_or_rectangles: np bool (n, ) or bool value
    """
    in_circles = points_in_circles(
        points,
        circles,
        clearance,
    )
    in_rectangles = points_in_rectangles(
        points,
        rectangles,
        clearance,
    )
    if type(points)==tuple:
        return bool(in_circles+in_rectangles)
    else:
        return (in_circles+in_rectangles).astype(bool)

def points_in_range(
    points,
    x_range,
    y_range,
    clearance=0,
):
    """
    - inputs:
        - points: np (n, 2) or tuple (2,)
        - x_range: (xmin, xmax)
        - y_range: (ymin, ymax)
        - clearance: scalar >= 0, clearance for boundary.
    - outputs:
        - np bool (n, ) or bool value
    """
    range_xywh = np.array([[x_range[0], y_range[0], x_range[1]-x_range[0], y_range[1]-y_range[0]]]) # (1,4)
    # shrink the range for boundary clearance
    return points_in_rectangles(
        points,
        range_xywh,
        clearance=-clearance,
    )
        
def points_validity(
    points,
    circle_obstacles,
    rectangle_obstacles,
    x_range,
    y_range,
    obstacle_clearance=0,
    range_clearance=0,
):
    """
    - inputs:
        - points: np (n, 2) or tuple (2,)
        - circle_obstacles: np (m, 3), (x, y, r), or None, which means no obstacles
        - rectangle_obstacles: np (m, 4), (x, y, w, h), or None, which means no obstacles
        - x_range: (xmin, xmax)
        - y_range: (ymin, ymax)
        - obstacle_clearance: scalar >= 0, clearance for obstacles.
        - range_clearance: scalar >= 0, clearance for range.
    - outputs:
        - np bool (n, ) or bool value
    """
    in_range = points_in_range(
        points,
        x_range,
        y_range,
        clearance=range_clearance,
    )
    in_circles = points_in_circles(
        points,
        circle_obstacles,
        clearance=obstacle_clearance,
    )
    in_rectangles = points_in_rectangles(
        points,
        rectangle_obstacles,
        clearance=obstacle_clearance,
    )
    # in range, and not in_circles, and not in_rectangles
    if type(points)==tuple:
        return bool(in_range*(1-in_circles)*(1-in_rectangles))
    else:
        return (in_range*(1-in_circles)*(1-in_rectangles)).astype(bool)