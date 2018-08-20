# required modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# two-dimesional version
def plot_mse_loss_surface_2d(fig, ax, x, y, v=0.0, l2=0.0, w1_range=(-2, 2), w2_range=(2, -2)):
    # create weight space
    n_w = 100
    w1 = np.linspace(w1_range[0], w1_range[1], num=n_w)  # weight 1
    w2 = np.linspace(w2_range[0], w2_range[1], num=n_w)  # weight 2
    ws_x, ws_y = np.meshgrid(w1, w2)
    cost_ws = np.zeros((n_w, n_w))  # initialize cost matrix

    # Fill the cost matrix for each combination of weights
    for i in range(n_w):
        for j in range(n_w):
            y_pred = ws_x[i, j] * ws_y[i, j] * x
            y_true = y
            cost_ws[i, j] = 0.5 * (y_true - y_pred)**2 + \
                0.5 * l2 * (ws_x[i, j]**2 + ws_y[i, j]**2) + 0.5 * v * (ws_x[i, j]*ws_y[i, j])**2

    # compute gradients
    dy, dx = np.gradient(cost_ws)

    # plot vector space
    skip = (slice(None, None, 5), slice(None, None, 5))
    im = ax.imshow(cost_ws, extent=[ws_x.min(), ws_x.max(
    ), ws_y.min(), ws_y.max()], cmap=cm.coolwarm)
    ax.quiver(ws_x[skip], ws_y[skip], -dx[skip], dy[skip], cost_ws[skip])
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('$Loss$', fontsize=15)

    ax.set_xlabel('$w_1$', fontsize=15)
    ax.set_ylabel('$w_2$', fontsize=15)

    # add saddle point
    ax.scatter(0, 0, label='Saddle point', c='red', marker='*')
    settings = (x, y, v, l2, w1_range, w2_range)

    return ax, settings


# three-dimensional version
def plot_mse_loss_surface_3d(ax, x, y, v=0.0, l2=0.0, w1_range=(-2, 2), w2_range=(2, -2), angle=30):
    # create weight space
    n_w = 100
    w1 = np.linspace(w1_range[0], w1_range[1], num=n_w)  # weight 1
    w2 = np.linspace(w2_range[0], w2_range[1], num=n_w)  # weight 2
    ws_x, ws_y = np.meshgrid(w1, w2)
    cost_ws = np.zeros((n_w, n_w))  # initialize cost matrix

    # Fill the cost matrix for each combination of weights
    for i in range(n_w):
        for j in range(n_w):
            y_pred = ws_x[i, j] * ws_y[i, j] * x
            y_true = y
            cost_ws[i, j] = 0.5 * (y_true - y_pred)**2 + \
                0.5 * l2 * (ws_x[i, j]**2 + ws_y[i, j]**2) + 0.5 * v * (ws_x[i, j]*ws_y[i, j])**2

    X = ws_x
    Y = ws_y
    Z = cost_ws

    # fourth dimention - colormap
    # create colormap according to x-value (can use any 50x50 array)
    color_dimension = np.log(Z-4.68459160059e-09) #Z  # change to desired fourth dimension
    minn, maxx = color_dimension.min(), color_dimension.max()
    #norm = Normalize(minn, maxx)
    norm = Normalize(-0.875466470014, 3.9608131696)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

    # plot
    ax.set_zlim(0, 50)
    ax.plot([0], [0], 'ro', c='red', marker='*', label='Saddle point')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=fcolors,
                    vmin=minn, vmax=maxx, shade=False, alpha=1)

    ax.set_xlabel('$w_1$', fontsize=20)
    ax.set_ylabel('$w_2$', fontsize=20)
    ax.set_zlabel('$Loss$', fontsize=20)

    settings = (x, y, v, l2, w1_range, w2_range)
    ax.view_init(angle, 10)

    return ax, settings


def plot_global_minimum_manifold_2d(ax, settings):
    # retieve cached settings
    x, y, v, l2, w1_range, w2_range = settings

    n_w = 1000
    man_w1 = np.linspace(w1_range[0], w1_range[1], num=n_w)
    man_w2 = np.linspace(w2_range[0], w2_range[1], num=n_w)
    man_ws_x, man_ws_y = np.meshgrid(man_w1, man_w2)
    loss = 0.5 * y *(1 - man_ws_x * man_ws_y * x)**2 + \
        0.5 * l2 * (man_ws_x**2 + man_ws_y**2) + 0.5 * v * (man_ws_x * man_ws_y)**2
    min_loss = np.min(loss)
    manifold_indices = loss < min_loss + 1e-5
    manifold_x = man_ws_x[manifold_indices]
    manifold_y = man_ws_y[manifold_indices]

    # plot manifold of global minima
    ax.scatter(manifold_y, manifold_x, s=0.1, c='cyan',
               label='Manifold of global minima')


def plot_global_minimum_manifold_3d(ax, settings):
    # retieve cached settings
    x, y, v, l2, w1_range, w2_range = settings

    n_w = 1000
    man_w1 = np.linspace(w1_range[0], w1_range[1], num=n_w)
    man_w2 = np.linspace(w2_range[0], w2_range[1], num=n_w)
    man_ws_x, man_ws_y = np.meshgrid(man_w1, man_w2)
    loss = 0.5 * y * (1 - man_ws_x * man_ws_y * x)**2 + \
        0.5 * l2 * (man_ws_x**2 + man_ws_y**2) + 0.5 * v * (man_ws_x*man_ws_y)**2
    min_loss = np.min(loss)
    manifold_indices = loss < min_loss + 1e-5
    manifold_x = man_ws_x[manifold_indices]
    manifold_y = man_ws_y[manifold_indices]

    pos = np.where(np.abs(np.diff(manifold_y)) >= 0.1)[0]+1
    x = np.insert(manifold_x, pos, np.nan)
    y = np.insert(manifold_y, pos, np.nan)

    # plot manifold of global minima
    ax.plot(y, x, c='cyan',
                label='Manifold of global minima')


def plot_optimiser_trajectory_2d(ax, weights, **kwargs):
    w1_vals = weights['w1']
    w2_vals = weights['w2']
    ax.plot(w1_vals, w2_vals, **kwargs)


def plot_optimiser_trajectory_3d(ax, settings, weights, **kwargs):
    x, y, v, l2, _, _ = settings
    w1_vals = np.array(weights['w1'])
    w2_vals = np.array(weights['w2'])
    loss = 0.5 * y * (1 - w1_vals * w2_vals * x)**2 + \
        0.5 * l2 * (w1_vals**2 + w2_vals**2) + 0.5 * v * (w1_vals*w2_vals)**2
    ax.plot(w1_vals, w2_vals, loss, **kwargs)


def plot_optimiser_trajectory(x, y, weights, dim='2d', angle=45, manifold=False, **kwargs):
    if dim == '3d':
        ax, settings = plot_mse_loss_surface_3d(x, y, angle=angle)
        if manifold:
            plot_global_minimum_manifold_3d(ax, settings)
        plot_optimiser_trajectory_3d(ax, settings, weights, **kwargs)
    else:
        ax, settings = plot_mse_loss_surface_2d(x, y)
        if manifold:
            plot_global_minimum_manifold_2d(ax, settings)
        plot_optimiser_trajectory_2d(ax, weights, **kwargs)


def plot_weight_norm(ax, weights, **kwargs):
    w1_vals = np.array(weights['w1'])
    w2_vals = np.array(weights['w2'])
    epochs = np.arange(0, len(w1_vals), 1)
    norms = np.sqrt(w1_vals**2 + w2_vals**2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Weight norm', fontsize=12)
    ax.plot(epochs, norms, linewidth=2.0, **kwargs)


def animate_optimiser_trajectory_2d(i, ax, weights, **kwargs):
    w1_vals = weights['w1']
    w2_vals = weights['w2']
    ax.plot(w1_vals[:i], w2_vals[:i], **kwargs)
    return ax


def animate_optimiser_trajectory_3d(i, ax, settings, weights, **kwargs):
    x, y, v, l2, _, _ = settings
    w1_vals = np.array(weights['w1'])
    w2_vals = np.array(weights['w2'])
    loss = 0.5 * y * (1 - w1_vals * w2_vals * x)**2 + \
        0.5 * l2 * (w1_vals**2 + w2_vals**2) + 0.5 * v * (w1_vals*w2_vals)**2
    ax.plot(w1_vals[:i], w2_vals[:i], loss[:i], **kwargs)
    return ax


def plot_optimiser_loss(x, y, v, l2, weights, **kwargs):
    loss = []
    epoch = np.arange(0, len(weights['w1']))
    for w1, w2 in zip(weights['w1'], weights['w2']):
        loss_val = 0.5 * y * (1 - w1 * w2 * x)**2 + 0.5 * l2 * (w1**2 + w2**2) + 0.5 * v * (w1 * w2)**2
        loss.append(loss_val)
    plt.plot(epoch, loss, **kwargs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


def plot_interpolated_trajectory_2d(ax, w1_a, w2_a, w1_b, w2_b, start=0, end=1, **kwargs):
    alpha = np.arange(start, end, 0.001)
    w1_path = []
    w2_path = []
    for a in alpha:
        ww1 = (1 - a) * w1_a + a * w1_b
        ww2 = (1 - a) * w2_a + a * w2_b
        w1_path.append(ww1)
        w2_path.append(ww2)
    ax.plot(w1_path, w2_path, **kwargs)


def plot_interpolated_trajectory_3d(ax, settings, w1_a, w2_a, w1_b, w2_b, start=0, end=1, **kwargs):
    x, y, _, _ = settings
    alpha = np.arange(start, end, 0.001)
    w1_path = []
    w2_path = []
    loss = []
    for a in alpha:
        ww1 = (1 - a) * w1_a + a * w1_b
        ww2 = (1 - a) * w2_a + a * w2_b
        loss_val = 0.5 * (y - ww1 * ww2 * x)**2 + 0.5 * l2 * (ww1**2 + ww2**2)
        loss.append(loss_val)
        w1_path.append(ww1)
        w2_path.append(ww2)
    ax.plot(w1_path, w2_path, loss, **kwargs)


def plot_interpolated_loss(x, y, w1_a, w2_a, w1_b, w2_b, start=0, end=1, **kwargs):
    alpha = np.arange(start, end, 0.001)
    interpolated_loss = []
    for a in alpha:
        ww1 = (1 - a) * w1_a + a * w1_b
        ww2 = (1 - a) * w2_a + a * w2_b
        loss_val = 0.5 * (y - ww1 * ww2 * x)**2 + 0.5 * l2 * (ww1**2 + ww2**2)
        interpolated_loss.append(loss_val)
    plt.plot(alpha, interpolated_loss, **kwargs)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Loss')


def plot_learning_dynamics(ax, weights, **kwargs):
    epoch = np.arange(0, len(weights['w1']))
    scores = []
    for w1, w2 in zip(weights['w1'], weights['w2']):
        scores.append(w1 * w2)
    ax.plot(epoch, scores, **kwargs)


def animate_learning_dynamics(i, ax, weights, y, **kwargs):
    n_epoch = len(weights['w1'])
    epoch = np.arange(1, n_epoch)
    scores = []
    for w1, w2 in zip(weights['w1'], weights['w2']):
        scores.append(w1 * w2)
    ax.set_xlim((1, n_epoch))
    ax.set_ylim((0, y))
    ax.set_xlabel('Epoch', fontsize=15)
    ax.set_ylabel('$w_2 \cdot w_1$', fontsize=15)
    ax.plot(epoch[:i], scores[:i], **kwargs)
    return ax


def animate_learning(weights, save=False, name='anim'):
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.5)

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(gs[0, :2], )
    ax2 = fig.add_subplot(gs[0, 2:], projection='3d')
    ax3 = fig.add_subplot(gs[1, 1:3])

    ax1, settings = plot_mse_loss_surface_2d(ax1, 1, 1)
    ax2, settings = plot_mse_loss_surface_3d(ax2, 1, 1, angle=60)
    plot_global_minimum_manifold_2d(ax1, settings)
    plot_global_minimum_manifold_3d(ax2, settings)

    def update(i):
        animate_optimiser_trajectory_2d(
            i, ax1, settings, weights, 'Gradient descent')
        animate_optimiser_trajectory_3d(
            i, ax2, settings, weights, 'Gradient descent')
        animate_learning_dynamics(i, ax3, weights, 1)

    # suncAnimation will call the 'update' function for each frame
    anim = FuncAnimation(fig, update, frames=100, interval=5, save_count=50)
    # HTML(anim.to_html5_video())

    if save:
        anim.save(name + '.gif', dpi=80, writer='imagemagick')
    plt.show()
