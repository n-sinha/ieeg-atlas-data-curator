import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_electrodes_with_rois(coords, roi_nums, cohort_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=roi_nums, cmap='tab20', s=10)
    plt.colorbar(sc, ax=ax, label='ROI Number')
    ax.set_title(f'{cohort_name} Electrode Positions by ROI')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()