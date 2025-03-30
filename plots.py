import matplotlib.pyplot as plt
import pandas as pd # type: ignore
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 1) Read the CSV file into a pandas DataFrame
data = pd.read_csv("results.txt")

# 2) Create a pivot table so that:
#    - The rows correspond to "Blocks"
#    - The columns correspond to "VectorSize"
#    - The cell values are "InclusiveTime_us"
#      (You can replace with "KernelTime_us" if you want kernel time instead.)
pivot = data.pivot_table(
    index="Blocks", 
    columns="VectorSize", 
    values="InclusiveTime_us", 
    aggfunc="mean"  # in case of duplicates
)

# 3) Sort the index and columns for a clean axis
pivot = pivot.sort_index().sort_index(axis=1)

# 4) Create the 2D grid for the surface plot
#    X will be the sorted VectorSize, Y will be the sorted Blocks
blocks_sorted = pivot.index.to_list()
vectors_sorted = pivot.columns.to_list()

X, Y = np.meshgrid(vectors_sorted, blocks_sorted)

# 5) Extract the Z values (InclusiveTime_us) from the pivot table
#    pivot is shaped (len(blocks), len(vectors)), so we do pivot.values
Z = pivot.values

# 6) Create a 3D figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# 7) Plot the surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

# 8) Label the axes
ax.set_xlabel("Vector Size (N)")
ax.set_ylabel("Blocks")
ax.set_zlabel("Total Time (us)")
ax.set_title("Blocks vs. N vs. Total Time")

# 9) Add a color bar
fig.colorbar(surf, shrink=0.5, aspect=10, label='Time (us)')

plt.tight_layout()
plt.show()
