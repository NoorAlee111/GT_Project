from itertools import combinations
from collections import Counter
import networkx as nx
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Assuming you have these functions and variables defined
from preprocessing import document_graphs, create_graph

# Function to compute the maximal common subgraph (MCS) between two graphs
def compute_mcs(G1, G2):
    # Convert graphs to edge sets
    edges1 = set(G1.edges())
    edges2 = set(G2.edges())

    # Compute the intersection of edges
    common_edges = edges1.intersection(edges2)

    # Create a new graph with common edges
    mcs_graph = nx.Graph(list(common_edges))

    return mcs_graph

# Function to compute distance between two graphs based on MCS
def compute_distance(G1, G2):
    mcs_graph = compute_mcs(G1, G2)
    return -len(mcs_graph.edges())

# Function to perform kNN classification
def knn_classify(test_graph, k, data):
    distances = []

    # Compute distance between test_graph and each training graph
    for train_id, train_graph in document_graphs.items():
        try:
            distance = compute_distance(test_graph, train_graph)
            distances.append((train_id, distance))
        except Exception as e:
            print(f"Error computing distance for train_id {train_id}: {e}")

    # Sort distances in ascending order
    distances.sort(key=lambda x: x[1])

    # Get the k-nearest neighbors
    neighbors = distances[:k]

    # Get categories of the neighbors
    neighbor_categories = [data.loc[i, 'Type'] for i, _ in neighbors]

    # Find the majority class
    majority_class = Counter(neighbor_categories).most_common(1)[0][0]

    return majority_class

# Read the data
try:
    data = pd.read_csv('ProcessedCSV.csv', encoding='latin1')
except FileNotFoundError:
    print("Error: 'merged_file.csv' not found. Please check the file path.")

# Create graph representations for test documents
test_documents = [
    create_graph(str(data.iloc[44]['Content'])),
    create_graph(str(data.iloc[28]['Content'])),
    create_graph(str(data.iloc[13]['Content']))
]

# Assuming you have actual categories for test documents
actual_categories = ['Science and Education', 'Food', 'Health and Fitness']  # Update with actual category labels

# Calculate evaluation metrics
try:
    predicted_categories = []
    for test_graph in test_documents:
        predicted_category = knn_classify(test_graph, k=3, data=data)
        predicted_categories.append(predicted_category)

    # Calculate evaluation metrics
    accuracy = accuracy_score(actual_categories, predicted_categories)
    precision = precision_score(actual_categories, predicted_categories, average='weighted', zero_division=1)
    recall = recall_score(actual_categories, predicted_categories, average='weighted')
    f1 = f1_score(actual_categories, predicted_categories, average='weighted')

    # Generate confusion matrix
    labels = sorted(set(actual_categories + predicted_categories))
    conf_matrix = confusion_matrix(actual_categories, predicted_categories, labels=labels)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)

    # Display results
    print("Predicted Categories:", predicted_categories)
    print("Actual Categories:", actual_categories)
    print("\nAccuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("\nConfusion Matrix:")
    print(conf_matrix_df)

except Exception as e:
    print(f"Error during evaluation: {e}")
