

This project implements an **AI-Driven Digital Twin** for structural heart stent analysis. By leveraging **GraphSAGE (SAGEConv)** architectures, the system predicts 3D Von Mises stress and displacement fields across complex metallic geometries based on design parameters like vertical link length.



## Project Overview
Traditional Finite Element Analysis (FEA) for medical devices is computationally expensive. This project utilizes **Geometric Deep Learning** to provide real-time structural feedback. The stent is partitioned into five vertical segments, each handled by a specialized GNN to maintain high spatial resolution and memory efficiency.

## Core Components

### 1. Data Extraction (`HT_Partitioned Main study_Displacement_0.2 to 0.4.py`)
This script automates the conversion of raw simulation data into a deep-learning-ready format.
* **Input**: Parses Nastran `input.bdf` (mesh) and `results.op2` (results) files using `pyNastran`.
* **Logic**: 
    * For cases $0.24\text{mm}$ and $0.26\text{mm}$, it extracts solid elements using **Property ID 3**.
    * For all other cases ($0.28\text{mm}$ to $0.44\text{mm}$), it utilizes **Property ID 4**.
* **Segmentation**: Generates 5 distinct PyTorch Geometric (`.pt`) files per simulation, representing vertical slices of the stent (0-20%, 20-40%, etc.).
* **Output**: Extracts nodal coordinates, edge connectivity, Von Mises stress, and 3D displacement vectors.



### 2. GNN Training (`HT_Partitioned Training_Displacement&Stress_0.2 to 1.py`)
The training pipeline utilizes **SAGEConv (GraphSAGE)** layers to perform inductive learning on the mesh graphs.
* **Why GraphSAGE?**: Unlike standard Graph Convolutional Networks (GCNs), GraphSAGE learns an aggregation function that generalizes to unseen topologies, making it ideal for predicting stress on stents with varying link lengths.
* **Architecture**: A multi-target GNN that simultaneously predicts scalar stress and vector displacement.
* **Scaling**: Features are standardized to millimeters to ensure numerical stability during the message-passing phase.

### 3. Interactive Dashboard (`app_2.py`)
A high-performance **Streamlit** application that acts as the primary interface for design exploration.
* **Segment Merging**: Dynamically loads and merges predictions from all 5 segment models into a unified 3D visualization.
* **Visual Controls**: "Full Black" high-contrast UI with features including:
    * **Z-Axis Stretching**: Exaggerates vertical scaling to better inspect link length variations.
    * **Peak Stress Marker**: Automatically locates and labels the node with the highest predicted stress.
    * **Safety Factor Calculation**: Real-time evaluation against 316L Stainless Steel yield strength.

