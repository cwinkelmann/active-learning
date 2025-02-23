import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load embeddings from CSV
df = pd.read_csv("embeddings_with_labels.csv", index_col=0)

# Extract feature vectors and labels
embeddings = df.iloc[:, :-1].values  # All columns except last (features only)
class_labels = df["class"].values  # Extract class labels

# Normalize embeddings
embeddings = StandardScaler().fit_transform(embeddings)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="random")
embeddings_2d = tsne.fit_transform(embeddings)

# Convert to DataFrame for visualization
df_tsne = pd.DataFrame(embeddings_2d, columns=["X", "Y"])
df_tsne["class"] = class_labels  # Append class labels

# Plot using Seaborn
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_tsne, x="X", y="Y", hue="class", palette="Set1", alpha=0.8, edgecolor="k")

plt.xlabel("t-SNE Dim 1")
plt.ylabel("t-SNE Dim 2")
plt.title("t-SNE Visualization of Image Embeddings")
plt.legend(title="Class")
plt.show()
