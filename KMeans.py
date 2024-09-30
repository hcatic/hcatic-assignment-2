import numpy as np

class KMeans:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.assignment = [-1 for _ in range(len(data))]
        self.manual_centroids = None
    
    def is_unassigned(self, i):
        """Check if a data point has not yet been assigned to a cluster."""
        return self.assignment[i] == -1

    def initialize_random(self):
        """Random initialization of centroids."""
        return self.data[np.random.choice(len(self.data), size=self.k, replace=False)]

    def initialize_farthest_first(self):
        """Farthest First initialization."""
        centers = [self.data[np.random.choice(len(self.data))]]
        for _ in range(1, self.k):
            dist = np.array([min([self.dist(c, p) for c in centers]) for p in self.data])
            next_center = self.data[np.argmax(dist)]
            centers.append(next_center)
        return np.array(centers)

    def initialize_kmeans_plus(self):
        """KMeans++ initialization."""
        centers = [self.data[np.random.choice(len(self.data))]]
        for _ in range(1, self.k):
            dist = np.array([min([self.dist(c, p) for c in centers]) for p in self.data])
            prob_dist = dist / dist.sum()
            next_center = self.data[np.random.choice(len(self.data), p=prob_dist)]
            centers.append(next_center)
        return np.array(centers)

    def initialize_manual(self):
        """Initialization using manual centroids."""
        if self.manual_centroids is None:
            raise ValueError("Manual centroids are not provided")
        if len(self.manual_centroids) != self.k:
            raise ValueError(f"Expected {self.k} centroids, but got {len(self.manual_centroids)}")
        return np.array(self.manual_centroids)

    def make_clusters(self, centers):
        """Assign each data point to the closest centroid."""
        for i in range(len(self.assignment)):
            dist = None
            for j in range(self.k):
                if self.is_unassigned(i):
                    self.assignment[i] = j
                    dist = self.dist(centers[j], self.data[i])
                else:
                    new_dist = self.dist(centers[j], self.data[i])
                    if new_dist < dist:
                        self.assignment[i] = j
                        dist = new_dist

    def compute_centers(self):
        """Recompute the centroids based on the current assignments."""
        centers = []
        for i in range(self.k):
            cluster = [self.data[j] for j in range(len(self.assignment)) if self.assignment[j] == i]
            if cluster:
                centers.append(np.mean(np.array(cluster), axis=0))
            else:
                centers.append(np.zeros(len(self.data[0])))
        return np.array(centers)
    
    def unassign(self):
        """Reset all assignments to -1."""
        self.assignment = [-1 for _ in range(len(self.data))]

    def are_diff(self, centers, new_centers):
        """Check if two sets of centroids are different."""
        for i in range(self.k):
            if self.dist(centers[i], new_centers[i]) != 0:
                return True
        return False

    def dist(self, x, y):
        """Compute Euclidean distance between two points."""
        return np.linalg.norm(x - y)

    def lloyds_step(self, current_centroids=None):
        """Perform a single step of Lloyd's algorithm."""
        if current_centroids is not None:
            centers = np.array(current_centroids)
        else:
            centers = self.initialize_random()

        self.make_clusters(centers)
        new_centroids = self.compute_centers()

        return new_centroids, self.assignment

    def lloyds_converge(self, current_centroids=None):
        """Run KMeans to convergence and return the final state."""
        if current_centroids is not None:
            centers = np.array(current_centroids)
        else:
            centers = self.initialize_random()

        self.make_clusters(centers)
        new_centroids = self.compute_centers()

        while self.are_diff(centers, new_centroids):
            self.unassign()
            centers = new_centroids
            self.make_clusters(centers)
            new_centroids = self.compute_centers()

        return new_centroids, self.assignment