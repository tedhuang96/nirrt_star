import numpy as np

def points_in_AABB_2d(points, aabb, clearance=0):
    """
    check whether 2D points are in 2D axis-aligned bounding boxes (rectangle obstacles)
    - inputs
        - points: np (n, 2) or tuple (2,)
        - aabb: np (m, 4), (x, y, w, h), or None, which means no obstacles
        - clearance: scalar
    - outputs:
        - in_aabb: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,2)
    if aabb is None:
        if points_type==tuple:
            in_aabb = False
        else:
            in_aabb = np.zeros(points.shape[0]).astype(bool)
        return in_aabb
    n, m = points.shape[0], aabb.shape[0]
    xp, yp = points[:,0:1], points[:,1:2] # (n, 1)
    xp, yp = xp*np.ones((1,m)), yp*np.ones((1,m)) # (n,m)
    xmin, ymin, w, h = aabb[:,0], aabb[:,1], aabb[:,2], aabb[:,3] # (m,)
    xmax, ymax = xmin+w+clearance, ymin+h+clearance
    xmin, ymin = xmin-clearance, ymin-clearance
    xmin, ymin, xmax, ymax = \
        xmin*np.ones((n,1)), ymin*np.ones((n,1)), xmax*np.ones((n,1)), ymax*np.ones((n,1)) # (n,m)
    in_aabb = (xmin<=xp)*(xp<=xmax)*(ymin<=yp)*(yp<=ymax) # (n,m)
    in_aabb = np.sum(in_aabb, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_aabb[0]
    return in_aabb

def points_in_ball_2d(points, ball, clearance=0):
    """
    check whether 2D points are in 2D balls (circle obstacles)
    - inputs
        - points: np (n, 2) or tuple (2,)
        - ball: np (m, 3), (x, y, r), or None, which means no obstacles
        - clearance: scalar
    - outputs:
        - in_ball: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,2)
    if ball is None:
        if points_type==tuple:
            in_ball = False
        else:
            in_ball = np.zeros(points.shape[0]).astype(bool)
        return in_ball
    n, m = points.shape[0], ball.shape[0]
    xp, yp = points[:,0:1], points[:,1:2] # (n, 1)
    xp, yp = xp*np.ones((1,m)), yp*np.ones((1,m)) # (n,m)

    xb, yb, r =  ball[:,0], ball[:,1], ball[:,2] # (m,)
    r = r+clearance

    xb, yb, r = \
        xb*np.ones((n,1)), yb*np.ones((n,1)), r*np.ones((n,1)) # (n,m)
    in_ball = ((xp-xb)**2+(yp-yb)**2)<=r**2 # (n,m)
    in_ball = np.sum(in_ball, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_ball[0]
    return in_ball


def points_in_AABB_3d(points, aabb, clearance=0):
    """
    check whether 3D points are in 3D axis-aligned bounding boxes (box obstacles)
    - inputs
        - points: np (n, 3) or tuple (3,)
        - aabb: np (m, 6), (x, y, z, w, h, d), or None, which means no obstacles
        - clearance: scalar
    - outputs:
        - in_aabb: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp,zp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,3)
    if aabb is None:
        if points_type==tuple:
            in_aabb = False
        else:
            in_aabb = np.zeros(points.shape[0]).astype(bool)
        return in_aabb
    n, m = points.shape[0], aabb.shape[0]
    xp, yp, zp = points[:,0:1], points[:,1:2], points[:,2:3] # (n, 1)
    xp, yp, zp = xp*np.ones((1,m)), yp*np.ones((1,m)), zp*np.ones((1,m)) # (n,m)
    xmin, ymin, zmin, w, h, d = aabb[:,0], aabb[:,1], aabb[:,2], aabb[:,3], aabb[:,4], aabb[:,5] # (m,)
    xmax, ymax, zmax = xmin+w+clearance, ymin+h+clearance, zmin+d+clearance
    xmin, ymin, zmin = xmin-clearance, ymin-clearance, zmin-clearance
    xmin, ymin, zmin, xmax, ymax, zmax = \
        xmin*np.ones((n,1)), ymin*np.ones((n,1)), zmin*np.ones((n,1)), xmax*np.ones((n,1)), ymax*np.ones((n,1)), zmax*np.ones((n,1)) # (n,m)
    in_aabb = (xmin<=xp)*(xp<=xmax)*(ymin<=yp)*(yp<=ymax)*(zmin<=zp)*(zp<=zmax) # (n,m)
    in_aabb = np.sum(in_aabb, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_aabb[0]
    return in_aabb


def points_in_ball_3d(points, ball, clearance=0):
    """
    check whether 3D points are in 3D balls (ball obstacles)
    - inputs
        - points: np (n, 3) or tuple (3,)
        - ball: np (m, 4), (x, y, z, r), or None, which means no obstacles
        - clearance: scalar
    - outputs:
        - in_ball: np bool (n, ) or bool value
    """
    points_type = type(points)
    if points_type==tuple:
        # (xp,yp,zp)
        points = np.array(points)[np.newaxis,:]
        assert points.shape==(1,3)
    if ball is None:
        if points_type==tuple:
            in_ball = False
        else:
            in_ball = np.zeros(points.shape[0]).astype(bool)
        return in_ball
    n, m = points.shape[0], ball.shape[0]
    xp, yp, zp = points[:,0:1], points[:,1:2], points[:,2:3] # (n, 1)
    xp, yp, zp = xp*np.ones((1,m)), yp*np.ones((1,m)), zp*np.ones((1,m)) # (n,m)

    xb, yb, zb, r =  ball[:,0], ball[:,1], ball[:,2], ball[:,3] # (m,)
    r = r+clearance

    xb, yb, zb, r = \
        xb*np.ones((n,1)), yb*np.ones((n,1)), zb*np.ones((n,1)), r*np.ones((n,1)) # (n,m)
    in_ball = ((xp-xb)**2+(yp-yb)**2+(zp-zb)**2)<=r**2 # (n,m)
    in_ball = np.sum(in_ball, axis=1).astype(bool) # (n,)
    if points_type==tuple:
        return in_ball[0]
    return in_ball