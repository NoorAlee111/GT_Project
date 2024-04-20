import os
from nltk.tokenize import word_tokenize
import networkx as nx
import matplotlib.pyplot as plt
from docx import Document
import csv

def text_to_directed_graph(text):
    words = word_tokenize(text)
    G = nx.DiGraph()
    unique_words = set(words)
    G.add_nodes_from(unique_words)
    for i in range(len(words)-1):
        word1 = words[i]
        word2 = words[i+1]
        if G.has_edge(word1, word2):
            G[word1][word2]['weight'] += 1
        else:
            G.add_edge(word1, word2, weight=1)
    return G

def plot_directed_graph(graph, filename):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    nx.draw(graph, pos, with_labels=True, font_weight='bold', arrows=True)
    labels = nx.get_edge_attributes(graph, 'weight')  # Get edge weights as labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)  # Draw edge labels
    plt.title(f'Graph for {filename}')
    plt.show()  # Display the plot

def save_graph_to_csv(graph, filename):
    csv_file = f"{filename.split('.')[0]}.csv"

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Source", "Target", "Weight"])

        # Write edges with weights to CSV
        for edge in graph.edges(data=True):
            source, target, weight = edge[0], edge[1], edge[2]['weight']
            writer.writerow([source, target, weight])

def read_text_from_docx(filename):
    doc = Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Specify the filenames relative to the current directory
for i in range(1, 13):
    filename = os.path.join(current_dir, f'Document_{i}.docx')

    try:
        # Read text from a DOCX file
        text = read_text_from_docx(filename)

        # Convert text to directed graph
        graph = text_to_directed_graph(text)

        # Plot the directed graph
        plot_directed_graph(graph, filename)

        # Save the graph as a single CSV file
        save_graph_to_csv(graph, filename)

        print(f"Processed: {filename}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
