import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def CreateSphere(center,r):
    u = np.linspace(0,2* np.pi,30)
    v = np.linspace(0,np.pi,30)
    x = np.outer(np.cos(u),np.sin(v))
    y = np.outer(np.sin(u),np.sin(v))
    z = np.outer(np.ones(np.size(u)),np.cos(v))
    x, y, z = r*x + center[0], r*y + center[1], r*z + center[2]
    return (x,y,z)

def draw_Spheres(ax,balls):
    for i in balls:
        (xs,ys,zs) = CreateSphere(i[0:3],i[-1])
        ax.plot_wireframe(xs, ys, zs, alpha=0.15,color="b")

def draw_block_list(ax, blocks ,color=None,alpha=0.15):
    '''
    drawing the blocks on the graph
    '''
    v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                 dtype='float')
    f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    n = blocks.shape[0]
    d = blocks[:, 3:6] - blocks[:, :3]
    vl = np.zeros((8 * n, 3))
    fl = np.zeros((6 * n, 4), dtype='int64')
    for k in range(n):
        vl[k * 8:(k + 1) * 8, :] = v * d[k] + blocks[k, :3]
        fl[k * 6:(k + 1) * 6, :] = f + k * 8
    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=alpha, linewidths=1, edgecolors='k')
        pc.set_facecolor(color)
        h = ax.add_collection3d(pc)
        return h

def obb_verts(obb):
    # 0.017004013061523438 for 1000 iters
    ori_body = np.array([[1,1,1],[-1,1,1],[-1,-1,1],[1,-1,1],\
                [1,1,-1],[-1,1,-1],[-1,-1,-1],[1,-1,-1]])
    # P + (ori * E)
    ori_body = np.multiply(ori_body,obb.E)
    # obb.O is orthornormal basis in {W}, aka rotation matrix in SO(3)
    verts = (obb.O@ori_body.T).T + obb.P
    return verts


def draw_obb(ax, OBB, color=None,alpha=0.15):
    f = np.array([[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [0, 1, 2, 3], [4, 5, 6, 7]])
    n = OBB.shape[0]
    vl = np.zeros((8 * n, 3))
    fl = np.zeros((6 * n, 4), dtype='int64')
    for k in range(n):
        vl[k * 8:(k + 1) * 8, :] = obb_verts(OBB[k])
        fl[k * 6:(k + 1) * 6, :] = f + k * 8
    if type(ax) is Poly3DCollection:
        ax.set_verts(vl[fl])
    else:
        pc = Poly3DCollection(vl[fl], alpha=alpha, linewidths=1, edgecolors='k')
        pc.set_facecolor(color)
        h = ax.add_collection3d(pc)
        return h


def draw_line(ax,SET,visibility=1,color=None):
    if SET != []:
        for i in SET:
            xs = i[0][0], i[1][0]
            ys = i[0][1], i[1][1]
            zs = i[0][2], i[1][2]
            line = plt3d.art3d.Line3D(xs, ys, zs, alpha=visibility, color=color)
            ax.add_line(line)

def visualization(initparams):
    if initparams.ind % 20 == 0 or initparams.done:
        V = np.array(list(initparams.V))
        # E = initparams.E
        Path = np.array(initparams.Path)
        start = initparams.env.start
        goal = initparams.env.goal
        # edges = E.get_edge()
        # generate axis objects
        ax = plt.subplot(111, projection='3d')
        #ax.view_init(elev=0.+ 0.03*initparams.ind/(2*np.pi), azim=90 + 0.03*initparams.ind/(2*np.pi))
        #ax.view_init(elev=0., azim=90.)
        ax.view_init(elev=90., azim=0.)
        #ax.view_init(elev=-8., azim=180)
        ax.clear()
        # drawing objects
        draw_Spheres(ax, initparams.env.balls)
        draw_block_list(ax, initparams.env.blocks)
        if initparams.env.OBB is not None:
            draw_obb(ax,initparams.env.OBB)
        draw_block_list(ax, np.array([initparams.env.boundary]),alpha=0)
        # draw_line(ax,edges,visibility=0.25)
        draw_line(ax,Path,color='r')
        if len(V) > 0:
            ax.scatter3D(V[:, 0], V[:, 1], V[:, 2], s=2, color='g',)
        ax.plot(start[0:1], start[1:2], start[2:], 'go', markersize=7, markeredgecolor='k')
        ax.plot(goal[0:1], goal[1:2], goal[2:], 'ro', markersize=7, markeredgecolor='k') 
        # adjust the aspect ratio
        set_axes_equal(ax)
        make_transparent(ax)
        # plt.xlabel('s')
        # plt.ylabel('y')
        plt.pause(0.0001)


def visualize_env_object(env):
    return visualize_env(
        env.boundary_no_clearance[3:].copy(),
        env.box_obstacles.copy(),
        env.balls_no_clearance.copy(),
        env.start.copy(),
        env.goal.copy(),
    )

def visualize_solution(env, path, vertices):
    return visualize_env(
        env.boundary_no_clearance[3:].copy(),
        env.box_obstacles.copy(),
        env.balls_no_clearance.copy(),
        env.start.copy(),
        env.goal.copy(),
        path,
        vertices,
    )


def visualize_env(env_dims, box_obstacles_np, ball_obstacles_np, start=None, goal=None, path=None, vertices=None):
    ax = plt.subplot(111, projection='3d')
    ax.view_init(elev=90., azim=0.)
    ax.clear()
    # drawing objects
    balls = ball_obstacles_np
    draw_Spheres(ax, balls)
    # box_obstacles [x,y,z,w,h,d]
    blocks = box_obstacles_np
    # import pdb; pdb.set_trace()
    blocks[:,3:] = blocks[:,:3]+blocks[:,3:] # [x,y,z,x+w,y+h,z+d]
    draw_block_list(ax, blocks)
    xmin, ymin, zmin = 0, 0, 0
    xmax, ymax, zmax = env_dims
    boundary = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
    draw_block_list(ax, np.array([boundary]),alpha=0)
    if path is not None:
        draw_line(ax,path,color='r')
    
    if vertices and len(vertices) > 0:
        vertices_np = np.array(vertices)
        ax.scatter3D(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2], s=2, color='g',)
    if start is not None and goal is not None:
        ax.plot(start[0:1], start[1:2], start[2:], 'go', markersize=7, markeredgecolor='k')
        ax.plot(goal[0:1], goal[1:2], goal[2:], 'ro', markersize=7, markeredgecolor='k')
    

    # ax.plot(start[0:1], start[1:2], start[2:], 'go', markersize=7, markeredgecolor='k')
    # ax.plot(goal[0:1], goal[1:2], goal[2:], 'ro', markersize=7, markeredgecolor='k') 
    # adjust the aspect ratio
    ax.set_xlim3d([xmin, xmax])
    ax.set_ylim3d([ymin, ymax])
    ax.set_zlim3d([zmin, zmax])
    ax.axis('off')
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz']); ax.set_box_aspect(np.ptp(limits, axis = 1))
    # set_axes_equal(ax)
    make_transparent(ax)
    # plt.xlabel('s')
    # plt.ylabel('y')
    # plt.pause(0.0001)

    # xmin, ymin, zmin = 0, 0, 0
    # xmax, ymax, zmax = env_dims
    return ax



def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def make_transparent(ax):
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

if __name__ == '__main__':
    pass