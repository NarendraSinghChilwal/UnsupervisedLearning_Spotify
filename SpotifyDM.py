#!/usr/bin/env python
# coding: utf-8

# In[230]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder


# In[231]:


df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")
df


# # BUSINESS UNDERSTANDING : 
# 
# Objective:
# To generate mood-based Spotify playlists by clustering songs based on audio features like danceability, energy, tempo, and liveness. The end goal is to create distinct clusters that represent different moods (e.g., "Energetic", "Relaxing", "Uplifting") for personalized playlist recommendations.
# 
# Key Questions:
# 1)What features contribute most to mood differentiation?
# 2)How many clusters (moods) should we create?

# In[ ]:





# # Exploratory Data Analysis
# 
# Examine distribution of features using histogram and boxplots

# In[234]:


#Checking the missing values 
df.isna().sum()


# In[235]:


print(df[df.isna().any(axis=1)])
#CHeck the missinig value


# In[ ]:





# In[236]:


#Visualizing data to check outliers

numerical_columns = ['popularity', 'duration_ms', 'danceability', 'energy', 'tempo', 'liveness']

# Adjusting the number of rows and columns to fit all 6 columns
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))  # 3 rows, 2 columns
for i, col in enumerate(numerical_columns):
    sns.histplot(df[col], ax=axes[i//2, i%2], kde=True)
    axes[i//2, i%2].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_columns])  # Correcting boxplot syntax
plt.title("Boxplot for Outlier Detection")
plt.show()


# # Data preparation:
# 
# 1)Log transformation
# 
# 2)Standard Scaling
# 
# 3)Visualizing effect of log transformation and Standard Scaling
# 

# In[238]:


#Dropping the row since no critical information is given which could help us find the data
cleaned_df = df.dropna().copy()
cleaned_df.isna().sum()
cleaned_df = cleaned_df.drop(columns=['Unnamed: 0'])
cleaned_df.reset_index(drop=True, inplace=True)
cleaned_df


# In[239]:


print(cleaned_df['track_genre'].unique())
print(cleaned_df['artists'].unique())


# In[ ]:





# In[240]:


#Since cardinality of artists is alot , so we can do either frequency encoding or target encoding
# One-hot encoding for track_genre
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_genres = pd.DataFrame(encoder.fit_transform(cleaned_df[['track_genre']]), columns=encoder.get_feature_names_out(['track_genre']))

# Frequency encoding for artists
artist_frequency = cleaned_df['artists'].value_counts().to_dict()
cleaned_df['artist_freq_encoded'] = cleaned_df['artists'].map(artist_frequency)

# Drop the old 'track_genre' and 'artists' columns
cleaned_df = cleaned_df.drop(columns=['track_genre', 'artists'])

# Concatenate the encoded columns
cleaned_df = pd.concat([cleaned_df, encoded_genres], axis=1)

# Now cleaned_df has one-hot encoded 'track_genre' and frequency encoded 'artists'


# In[241]:


cleaned_df


# In[242]:


print(cleaned_df.isna().sum())


# In[243]:


#Data is ready to scale so keeping the record on irrelevant columns in a separate dataframe

track_info = cleaned_df[['track_id','track_name','album_name']]
cleaned_df = cleaned_df.drop(columns = ['track_id','track_name','album_name'], axis=1)


# In[244]:


#Since, popularity, duration_ms, and liveness are right skewed, we will do normalization(log transformation) to reduce the impact of extreme values

cleaned_df['popularity'] = np.log1p(cleaned_df['popularity'])
cleaned_df['duration_ms'] = np.log1p(cleaned_df['duration_ms'])
cleaned_df['liveness'] = np.log1p(cleaned_df['liveness'])


# In[ ]:





# In[245]:


#For danceability, energy and tempo , we can do standard scaling as the distribution looks normal

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cleaned_df[['danceability', 'energy', 'tempo']] = scaler.fit_transform(cleaned_df[['danceability', 'energy', 'tempo']])


# In[ ]:





# In[246]:


#Visualizing again to check the effect of normalization and scaling

numerical_columns = ['popularity', 'duration_ms', 'danceability', 'energy', 'tempo', 'liveness']

# Adjusting the number of rows and columns to fit all 6 columns
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))  # 3 rows, 2 columns
for i, col in enumerate(numerical_columns):
    sns.histplot(cleaned_df[col], ax=axes[i//2, i%2], kde=True)
    axes[i//2, i%2].set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

# Boxplot to detect outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=cleaned_df[numerical_columns]) 
plt.title("Boxplot for Outlier Detection")
plt.show()


# # ML MODEL
# 
# Elbow method to find optimal value of k

# In[248]:


# Try a range of k values
inertia = []
k_range = range(1, 11)  # Try k from 1 to 10

X = cleaned_df[numerical_columns].values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X) 


for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_pca)  # Use the PCA-transformed data
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.show()


# # 1) K-means clustering

# In[250]:


#Testing with k=6

# Apply KMeans clustering with KMeans++ initialization
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=23316144)
kmeans.fit(X_pca)

# Get the cluster centers and labels
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the KMeans clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')  # Red crosses for centroids
plt.title('KMeans Clusters of Songs')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()


# In[251]:


silhouette_avg = silhouette_score(X_pca, labels)
db_score = davies_bouldin_score(X_pca, labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Score: {db_score}")


# In[252]:


# Apply KMeans clustering with KMeans++ initialization
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=23316144)
kmeans.fit(X_pca)

# Get the cluster centers and labels
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the KMeans clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', marker='o')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')  # Red crosses for centroids
plt.title('KMeans Clusters of Songs')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()


# In[253]:


silhouette_avg = silhouette_score(X_pca, labels)
db_score = davies_bouldin_score(X_pca, labels)

print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Score: {db_score}")


# # Building playlist
# 
# k = 5 is optimal so will now create a playlist and remove unrelated columns

# In[255]:


# We already have the cluster labels from KMeans
cleaned_df['Cluster'] = kmeans.labels_  # Add the cluster labels to the DataFrame
cleaned_df['Cluster'].unique()


# In[256]:


cleaned_df.columns


# # FORMING CLUSTER BASED ON FEATURE GRAPH

# In[258]:


# Define the audio features to analyze
audio_features = ['popularity', 'danceability', 'energy', 'speechiness', 'acousticness']

# Group by 'Cluster' and calculate the mean for each audio feature
cluster_means = cleaned_df.groupby('Cluster')[audio_features].mean()

# Plot a grouped bar plot
cluster_means.plot(kind='bar', figsize=(12, 8), color=['skyblue', 'lightgreen', 'salmon', 'orange', 'purple'])

# Add titles and labels
plt.title('Cluster-wise Average Values of Audio Features', fontsize=16)
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Average Value', fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.legend(title='Audio Features', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()


# In[259]:


"""
Based on the bar graph , we can define cluster as followings:

    0: "Acoustic & Mellow",
    1: "Popular Hits",
    2: "Upbeat Instrumentals",
    3: "Live Performances & Spoken",
    4: "High-Energy Rhythms",

"""

cluster_genres = {
    0: "Acoustic & Mellow",
    1: "Popular Hits",
    2: "Upbeat Instrumentals",
    3: "Live Performances & Spoken",
    4: "High-Energy Rhythms",
}



# Define the function to label clusters based on observations
def label_clusters(row):
    """
    Assign descriptive labels to clusters based on their characteristics.
    Adjust the labels based on feature analysis and observations.
    """
    if row['Cluster'] == 0:
        return 'Acoustic & Mellow' 
    elif row['Cluster'] == 1:
        return 'Popular Hits'  
    elif row['Cluster'] == 2:
        return 'Upbeat Instrumentals'  
    elif row['Cluster'] == 3:
        return 'Live Performances & Spoken' 
    elif row['Cluster'] == 4:
        return 'High-Energy Rhythms'  
    else:
        return 'Unknown Cluster'  # Default label for any unexpected clusters


# Apply the function to your cleaned dataset to create a new 'Cluster_Label' column
cleaned_df['Cluster_Label'] = cleaned_df.apply(label_clusters, axis=1)

# Display the cleaned dataframe with the new cluster labels
print(cleaned_df[['Cluster', 'Cluster_Label']].head())

# Optionally, you can also count how many songs are in each label to verify the distribution
print(cleaned_df['Cluster_Label'].value_counts())


# In[260]:


merged_df = pd.concat([track_info, cleaned_df], axis=1)

playlist_df = merged_df[['track_name', 'album_name', 'Cluster_Label']]
playlist_df


# In[261]:


# Group the songs by 'Cluster_Label'
grouped_playlists = playlist_df.groupby('Cluster_Label')

# Print only the first 5 songs in each cluster
for cluster_label, group in grouped_playlists:
    print(f"Playlist for {cluster_label}:")
    print(group[['track_name', 'album_name']].head(5).to_string(index=False))  # Show only the top 5
    print("\n" + "="*50 + "\n")


# In[ ]:




