# Customer Segmentation using K-Means Clustering

## Project Overview
This project implements customer segmentation using K-Means clustering on a dataset of mall customers. The objective is to identify distinct customer groups based on their annual income and spending score.

## Key Features
- **Data Preprocessing:** 
  - Loaded customer data from a CSV file.
  - Standardized the features (annual income and spending score) using `StandardScaler` to improve clustering performance.
  
- **Elbow Method for Optimal Clusters:** 
  - Employed the Elbow Method to determine the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS) against the number of clusters.

- **K-Means Clustering:** 
  - Applied K-Means clustering with the optimal number of clusters (determined to be 5) to segment customers.
  
- **Visualization:** 
  - Visualized the clusters and their centroids in a scatter plot, effectively illustrating customer segmentation based on income and spending habits.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn

## Getting Started
To run this project, you will need:
- Python 3.x
- Required libraries (listed in `requirements.txt`)

### Installation
Clone this repository and install the necessary libraries:
```bash
pip install -r requirements.txt
