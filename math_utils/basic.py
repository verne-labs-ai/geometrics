import torch
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator


# batch*n
def normalize_vector( v, return_mag =False):
    batch=v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))# batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch,1).expand(batch,v.shape[1])
    v = v/v_mag
    if(return_mag==True):
        return v, v_mag[:,0]
    else:
        return v

# u, v batch*n
def cross_product( u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:,1]*v[:,2] - u[:,2]*v[:,1]
    j = u[:,2]*v[:,0] - u[:,0]*v[:,2]
    k = u[:,0]*v[:,1] - u[:,1]*v[:,0]
        
    out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)#batch*3
        
    return out


def geodesic_distance(angle1, angle2):
    """
    Compute the geodesic distance between two angles on a circle.
    
    Parameters:
    angle1 (float): First angle in radians.
    angle2 (float): Second angle in radians.
    
    Returns:
    float: Geodesic distance between the two angles in radians.
    """
    # Calculate the absolute difference between the two angles
    diff = np.abs(angle1 - angle2)

    # Ensure the distance is within the range [0, π]
    geodesic_dist = np.minimum(diff, 2 * np.pi - diff)

    return geodesic_dist


def cluster_3d_points(points, max_k: int = 10, visualize: bool = False):
    """
    Cluster a list of 3D points into an optimal number of clusters (between 1 and 10)
    using the elbow method based on KMeans inertia (loss).

    Parameters:
        points (list or array-like): List of 3D points, e.g., [(x1,y1,z1), (x2,y2,z2), ...]

    Returns:
        optimal_k (int): The chosen optimal number of clusters.
        labels (ndarray): Cluster labels assigned to each point.
    """
    # Convert input to a NumPy array for easier processing
    points_array = np.array(points)

    inertia_values = []
    models = []

    assert len(points_array) >= max_k, "Not enough points to cluster"
    assert max_k >= 1, "max_k must be at least 1"
    
    # Evaluate KMeans for k from 1 to max_k
    for k in range(1, max_k+1):
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(points_array)
        inertia_values.append(model.inertia_)
        models.append(model)

    # Try to detect the elbow using the KneeLocator from the kneed library
    kn = KneeLocator(range(1, max_k+1), inertia_values, curve='convex', direction='decreasing')
    if visualize:
        kn.plot_knee()
        plt.show()
    optimal_k = kn.knee
    if optimal_k is None:
        # If no clear knee is found, fallback to a default value
        optimal_k = 1

    # Get the labels from the best model (index is optimal_k-1)
    best_model = models[optimal_k - 1]
    return optimal_k, best_model.labels_
