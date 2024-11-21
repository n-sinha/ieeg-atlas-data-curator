import matplotlib.pyplot as plt

def plot_electrode_mappings(electrode_coords, roi_nums, title):
    """
    Plot electrode coordinates colored by ROI numbers.
    
    Args:
        electrode_coords (ndarray): Coordinates of electrodes.
        roi_nums (ndarray): ROI numbers for each electrode.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(electrode_coords[:, 0], electrode_coords[:, 1], c=roi_nums, cmap='jet', s=20)
    plt.colorbar(scatter, label='ROI Number')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.tight_layout()
    plt.show()