# FILENAME: RS-sampling.py
"""
Remote Sensing sampling

This code block does the following:
- 
- 
"""

# pip install tifffile rasterio matplotlib numpy scipy scikit-learn networkx pyproj

import tifffile
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import pyproj

# Open and inspect the GeoTIFF file
# --------------------
tif_filename = './ndvi-2022-01-01_a_2022-01-15.tif'
# tif_filename = './ndvi-2022-01-01_a_2022-01-15.tif'
# tif_filename = './bandas.tif'

# Read TIFF file
with tifffile.TiffFile(tif_filename) as tif:
    # Get first image from TIFF file
    image = tif.pages[0].asarray()

    # Display basic info
    print(f"Shape: {image.shape}")
    print(f"Data type: {image.dtype}")

    # Display all images in TIFF file
    for page in tif.pages:
        print(page.asarray())

# Open and plot the GeoTIFF file
# --------------------
tif_filename = './geoTiffs-2018_a_2023/2019-01-01_a_2019-01-31.tif'
# tif_filename = './ndvi-2022-01-01_a_2022-01-15.tif'
# tif_filename = './bandas.tif'

# use rasterio to plot tiff band information
with rasterio.open(tif_filename) as raster:
    # set band
    band_n = 1 # starting from 1
    band = raster.read(band_n)

    # Show the image
    plt.imshow(band, cmap='viridis')
    plt.colorbar()
    plt.show()
print(band.shape)

# Compute band histogram to inspect distribution
# --------------------
# get max and min values from band1.ravel()
min_n = band[np.isfinite(band)].min()
max_n = band[np.isfinite(band)].max()
bins_n = 50
# make a histogram from the data array and plot it
plt.hist(band.ravel(), bins=bins_n, range=(min_n, max_n));

# Show image, identifying pixels with NDVI values in a given percentile interval
# --------------------
# Set percentile thresholds, must be between 0 and 100.
index_min, index_max = 40, 60

if(index_max < 0 or index_max > 100): 
    print("index_max must be between 0 and 100") 
if(index_min < 0 or index_min > 100): 
    print("index_min must be between 0 and 100")

# compute quantiles corresponding to 
q1, q2 = np.nanpercentile(band.ravel()[~np.isneginf(band.ravel())], [index_min, index_max])
# select values from band between quantiles
candidate_set = np.ma.masked_outside(band, q1, q2)
print(f"Selected set contains {candidate_set.count()} pixels inside a {candidate_set.shape} shape.")
# candidate_set.shape

# plot band in a 2d image, masking values between q1 and q2, every point is reddish
plt.imshow(band, cmap="viridis")
plt.colorbar()
plt.imshow(candidate_set, cmap="autumn", vmax=1, vmin=1)
plt.show()

print(f"Percentile interval is: [",round(q1, 2), ",", round(q2, 2), "]")


## review second part of this block

import pyproj

def get_xy_from_index(array, raster, index):
    # get x and y from the index of a pixel-pair vector
    # array must conform to the interpixel_d of a rasterio open() function
    if(index > array.shape[0]-1 or index < 0):
        print("index must be between 0 and array.count()")
    else:
        row = array[index][0]
        col = array[index][1]
        return(raster.xy(row, col))

def d_ij(array, raster, i, j):
    # Define the geodetic projection
    geod = pyproj.Geod(ellps='WGS84')
    xi = get_xy_from_index(array, raster, i)[0]
    xj = get_xy_from_index(array, raster, j)[0]
    yi = get_xy_from_index(array, raster, i)[1]
    yj = get_xy_from_index(array, raster, j)[1]
    _, _, h_distance = geod.inv(xi, yi, xj, yj)
    e_distance = ((xj - xi)**2 + (yj - yi)**2)**0.5
    print("d_ij:", round(xi, 4), round(yi, 4), round(xj, 4), round(yj, 4), "haversine: ", round(h_distance, 4), "euclidian", round(e_distance*1e5, 4), "error", round(abs(h_distance-e_distance*1e5)/h_distance*100, 2))
    return(h_distance)
    # return(h_distance)


candidate_list = np.argwhere(candidate_set)
candidate_list_n = candidate_list.shape[0]

interpixel_d= np.zeros((candidate_list_n, candidate_list_n))
mean_d, median_d = np.zeros(candidate_list_n), np.zeros(candidate_list_n)

for n in range(0, candidate_list_n):
    for m in range(n, candidate_list_n):
        interpixel_d[n][m] = d_ij(candidate_list, raster, n, m)

    # compute median and mean of interpixel distances, es metrics for describing pixels
    mean_d[n] = np.mean(interpixel_d[n])
    median_d[n] = np.median(interpixel_d[n])
# computed indices of sorted mean array and print them
sorted_mean_indices = np.argsort(mean_d)
sorted_median_indices = np.argsort(median_d)

print(interpixel_d)

# Find the index of the maximum element in the array
max_index = np.argmax(interpixel_d)

# Convert the max_index to row and column indices for the largest distance
row_index, col_index = np.unravel_index(max_index, interpixel_d.shape)
print(f"Maximum distance is {interpixel_d[row_index][col_index]} between pixels {candidate_list[row_index]} and {candidate_list[col_index]}")


k=6
## using only mean or median information doesn't work very well as tends to form clusters
# k_samples = candidate_list[sorted_mean_indices[-k:]]
# candidate_list[median_d[sorted_median_indices[-k]]]

# use most distant pairs and build a sample from those
sorted_indices = np.argsort(interpixel_d, axis=None)
print(f"sorted indices ", sorted_indices[-k:])
k_largest_indices = np.asarray(np.unravel_index(sorted_indices[-k:], interpixel_d.shape))
# print(f"k_largest_indices", k_largest_indices.shape)
# print(f"k_largest_indices", k_largest_indices[0, :], k_largest_indices[1, :])
# print(f"k candidate_list", candidate_list[k_largest_indices[0, :]], "another ", candidate_list[k_largest_indices[1, :]])

candidate_list[k_largest_indices[0, :]]
kplus_samples = np.vstack((candidate_list[k_largest_indices[0, :]], candidate_list[k_largest_indices[1, :]]))
k_samples = np.unique(kplus_samples, axis=0)[-k:]
# print(k_largest_indices, k_largest_indices[:, 0])

# extract the x and y coordinates
k_samples_x = k_samples[:, 1]
k_samples_y = k_samples[:, 0]

plt.imshow(band, cmap="viridis")
plt.colorbar()
plt.scatter(k_samples_x, k_samples_y, c='red', marker='o')
plt.show()

# Kmeans
from sklearn.cluster import KMeans
# Cluster the elements into k clusters using KMeans
k = k

kmeans = KMeans(n_clusters=k, random_state=0).fit(candidate_list)

# Get the labels of the clusters
labels = kmeans.labels_

# Select one random element from each cluster
random_elements = []
for i in range(k):
    cluster_indices = np.where(labels == i)[0]
    random_index = np.random.choice(cluster_indices)
    random_element = candidate_list[random_index]
    random_elements.append(random_element)

# Convert the list of random elements to a (k, 2) ndarray
k_samples = np.array(random_elements)

# Print the random elements from each cluster
print(k_samples)
# extract the x and y coordinates
k_samples_x = k_samples[:, 1]
k_samples_y = k_samples[:, 0]

plt.imshow(band, cmap="viridis")
plt.colorbar()
plt.scatter(k_samples_x, k_samples_y, c='red', marker='o')
plt.show()

# TSP solver - SA
import networkx as nx

nodes = k_samples
N = k_samples.shape[0]

# Create a fully connected graph
G = nx.Graph()
for i in range(N):
    for j in range(i+1, N):
        node1 = tuple(nodes[j])
        node2 = tuple(nodes[i])
        distance = d_ij(candidate_list, raster, i, j)
        G.add_edge(node1, node2, weight=distance)
        G[node1][node2]['weight']

# Print the edges and weights of the graph
# for edge in G.edges(data=True):
#     print(edge)

# Compute the shortest path that passes through all nodes
# start_node = tuple(k_samples[np.random.randint(N)])
# start_node

tsp_path = nx.algorithms.approximation.traveling_salesman_problem(G, weight='weight', cycle=True)

# Print the shortest path
print(tsp_path)

# Sum the weights of the edges in the TSP path
total_weight = 0
for i in range(len(tsp_path)-1):
    node1 = tsp_path[i]
    node2 = tsp_path[i+1]
    print(node1, node2, G[node1][node2]['weight'])
    total_weight += G[node1][node2]['weight']
# total_weight += G[tsp_path[-1]][tsp_path[0]]['weight']

# Print the total weight
print(total_weight)

# Print the random elements from each cluster
print("sample", k_samples.shape)
# extract the x and y coordinates
k_samples_x = k_samples[:,1]
k_samples_y = k_samples[:,0]

plt.imshow(band, cmap="viridis")
plt.colorbar()
plt.scatter(k_samples_x, k_samples_y, c='red', marker='o')
plt.show()

# TSP solver - Bruteforce
# Convert the list of random elements to a (k, 2) ndarray
k_samples = np.array(random_elements)

# Print the random elements from each cluster
# print(k_samples, k_samples.shape)

# extract the x and y coordinates
k_samples_x = k_samples[:, 1]
k_samples_y = k_samples[:, 0]

cycle = nx.algorithms.approximation.traveling_salesman_problem(G, weight='distance')
# cycle = nx.algorithms.approximation.simulated_annealing_tsp(G, init_cycle="greedy", source=cycle[6], weight='distance', temp=1e8, alpha=0.9999999, max_iterations=1e8)

# Function to calculate the total weight of a path
def path_weight(G, path):
    weight = 0
    for i in range(len(path) - 1):
        weight += G[path[i]][path[i + 1]]['weight']
        print(weight, path[i], path[i+1])
    return weight

import itertools
# Brute-force TSP solver
def brute_force_tsp(G, start):
    min_weight = float("inf")
    min_path = []
    vertices = list(G.nodes())
    vertices.remove(start)
    
    # Generate all possible permutations
    for perm in itertools.permutations(vertices):
        path = [start] + list(perm) + [start]
        # path = list(perm) + [start]
        weight = path_weight(G, path)
        print(weight, path)
        
        if weight < min_weight:
            min_weight = weight
            min_path = path
            
    return min_path, min_weight

# Given an initial i0, j0, get the closest node and start from there
init_pix_n = 250
init_pix = np.argwhere(np.ma.masked_invalid(band))[init_pix_n]
i0, j0 = init_pix

nodes_array = np.array([node for node in G.nodes()])
print("nodes_array", nodes_array)
nodes_array = np.insert(nodes_array, 0, [i0, j0], axis=0)
print("nodes_array", nodes_array)

min_d = float("inf")
min_index = 0
for i in range(nodes_array.shape[0]):
    if min_d > d_ij(nodes_array, raster, 0, i) and i>0:
        min_d = d_ij(nodes_array, raster, 0, i)
        # print("min dist", i, "is", min_d, "node", nodes_array[i])
        min_index = i

print("initial position is at", i0, j0)
print(f"closest to sample number", min_index, "at", nodes_array[min_index], "distance", min_d)

# Solve TSP - brute force approach
start_vertex = cycle[min_index]
print("cycle", cycle)
cycle, total_distance = brute_force_tsp(G, start_vertex)

# print(interpixel_d)
print(f"Secuencia de muestras: ", cycle, "y distancia total: ", total_distance + min_d)

# compute the total distance from path
total_distance = 0
for i in range(len(cycle)-1):
    node1 = cycle[i]
    node2 = cycle[i+1]
    total_distance += G[node1][node2]['weight']
print(f"Total distance is {total_distance}")

plt.imshow(band, cmap="viridis")
plt.colorbar()
plt.scatter(k_samples_x, k_samples_y, c='red', marker='o')
plt.scatter(j0, i0, c='blue', marker='o')
plt.annotate(f'({j0},{i0})', (j0, i0))

enumerate(k_samples)
# add labels to the points
for i, p in enumerate(k_samples):
    plt.annotate(f'({k_samples_x[i]},{k_samples_y[i]})', (k_samples_x[i], k_samples_y[i]))

# Plot the TSP path
for i in range(len(cycle)-1):
    node1 = cycle[i]
    node2 = cycle[i+1]
    plt.plot([node1[1], node2[1]], [node1[0], node2[0]], 'k-')
plt.plot([init_pix[1], nodes_array[min_index][1]], [init_pix[0], nodes_array[min_index][0]], 'b--')
plt.show()

