
import faiss
import numpy as np

#kmeans
def self_clustering_kmeans(embeddings,targets):
    N,D = embeddings.size()
    kmeans = faiss.Kmeans()
    kmeans.train(embeddings)
    dists, ids = kmeans.index.search(embeddings,1)


def knn_w_centroids(centroids,embeddings,targets,K):
    N,D = embeddings.size()

    #normalize feature to cos sim
    index = faiss.IndexFlatL2(D)
    # gpu_index = faiss.index_cpu_to_all_gpus(index)
    index.add(embeddings)

    dists, ids = index.search(centroids,k=K)

    

    
#knn
# def 