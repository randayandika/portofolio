#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation using K-Means Clustering

# ## Load & Preprocessing each dataset

# ### 1. Data Customer

# In[1]:


import pandas as pd
import numpy as np

# load dataset 
path = 'Case Study - Customer.csv'
customer = pd.read_csv(path, index_col=0, sep=";").reset_index()
customer.head(5)


# In[2]:


customer.info()


# In[3]:


customer.isnull().sum()


# In[4]:


mode = customer["Marital Status"].mode()[0]
customer["Marital Status"].fillna(mode, inplace=True)


# In[5]:


customer.isnull().sum()


# In[6]:


customer.duplicated().sum()


# In[7]:


customer.drop_duplicates(inplace=True)


# In[8]:


customer["Income"] = customer["Income"].str.replace(",", ".").astype(float)


# In[9]:


customer.head(5)


# ### 2. Data Product

# In[10]:


# load dataset 
path = 'Case Study - Product.csv'
product = pd.read_csv(path, index_col=0, sep=";").reset_index()
product.head(10)


# In[11]:


product.info()


# ### 3. Data Store

# In[12]:


# load dataset 
path = 'Case Study - Store.csv'
store = pd.read_csv(path, index_col=0, sep=";").reset_index()
store.head(10)


# In[13]:


store.info()


# In[14]:


store["Latitude"] = store["Latitude"].str.replace(",", ".").astype(float)
store["Longitude"] = store["Longitude"].str.replace(",", ".").astype(float)

store.head(5)


# ### 4. Data Transaksi

# In[15]:


# load dataset 
path = 'Case Study - Transaction.csv'
transaction = pd.read_csv(path, index_col=0, sep=";").reset_index()
transaction.head(5)


# In[16]:


transaction.info()


# In[17]:


transaction.isnull().sum()


# In[18]:


transaction.duplicated().sum()


# ## Merge Data & Create Master Data

# In[19]:


transaction_store = transaction.merge(store, on='StoreID', how='inner')
transaction_store_customer = transaction_store.merge(customer, on='CustomerID', how='inner')
df = transaction_store_customer.merge(product, on='ProductID', how='inner')

df.head(10)


# In[20]:


df.info()


# In[21]:


categorical = df.select_dtypes(exclude=[np.number])
categorical.columns


# In[22]:


numerical = df.select_dtypes(include=[np.number])
numerical.columns


# ### Membuat data baru untuk clustering, yaitu groupby by customerID lalu yang di aggregasi adalah :
# * Transaction id count
# * Qty sum
# * Total amount sum
# 

# In[23]:


# Group and aggregate the data
new_data = df.groupby('CustomerID').agg({'TransactionID': 'count', 'Qty': 'sum', 'TotalAmount': 'sum'}).reset_index()

# Display the first 10 rows of the new_data DataFrame
new_data.head(10)


# In[24]:


new_data.describe()


# ## Handling Outliers

# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns

# Create a boxplot
plt.figure(figsize=(8, 6))  # Set the size of the figure
sns.boxplot(data=new_data[['TransactionID', 'Qty','TotalAmount']])  # Create the boxplot
plt.title('Boxplot of Customer Data')  # Set the title of the plot
plt.xlabel('Variable')  # Set the label for the x-axis
plt.ylabel('Value')  # Set the label for the y-axis
plt.show()  # Show the plot


# In[26]:


print(f'Jumlah Baris Sebelum Outlier Dihapus: {len(new_data)}')
filtered_entries = np.array([True] * len(new_data))
for col in['Qty','TotalAmount']:
    
    q1=new_data[col].quantile(0.25)
    q3=new_data[col].quantile(0.75)
    iqr=q3-q1

    min_IQR = q1 - (1.5 * iqr)
    max_IQR = q3 + (1.5 * iqr)

    filtered_entries=((new_data[col]>=min_IQR) & (new_data[col]<=max_IQR)) & filtered_entries
    new_data=new_data[filtered_entries]

print(f'Jumlah Baris Sebelum Outlier Dihapus: {len(new_data)}')


# ## Scatter Plot Qty & TotalAmount

# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,8))
sns.scatterplot(data=new_data, x='Qty', y='TotalAmount')


# ## Clustering Process

# In[28]:


# Define the list of feature names
feats = ['TransactionID', 'Qty', 'TotalAmount']

# Select the specified columns as feature values
X = new_data[feats].values

# Import the StandardScaler class
from sklearn.preprocessing import StandardScaler

# Standardize the feature values
X_std = StandardScaler().fit_transform(X)

# Create a new DataFrame with standardized features
new_df = pd.DataFrame(data=X_std, columns=feats)

# Generate descriptive statistics for the new DataFrame
new_df.describe()


# ##  Elbow Method: Inertia 

# In[29]:


## Mencari N CLuster yang pas

from sklearn.cluster import KMeans  # Import the KMeans class
inertia = []  # Initialize an empty list to store inertia values

# Loop through different numbers of clusters
for i in range(2, 11):
  kmeans = KMeans(n_clusters=i, random_state=0)  # Create a KMeans instance with the specified number of clusters
  kmeans.fit(X_std)  # Fit the KMeans model to the standardized feature values
  nilai_inertia = kmeans.inertia_  # Get the inertia value of the clustering result
  print('iterasi ke-', i, 'dengan nilai inertia: ', nilai_inertia)  # Print the iteration number and inertia value
  inertia.append(kmeans.inertia_)  # Append the inertia value to the list


# In[30]:


plt.figure(figsize=(7, 5))  # Set the size of the figure

# Plot the line plot of inertia values
sns.lineplot(x=range(2, 11), y=inertia, color='#000087', linewidth=4)

# Plot the scatter plot of inertia values
sns.scatterplot(x=range(2, 11), y=inertia, s=300, color='#800000', linestyle='--')


# In[31]:


# visualisasi innertia vs k dengan parameter distortion
from yellowbrick.cluster import KElbowVisualizer

# fit model
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,11), metric='distortion', timings=True, locate_elbow=True)
visualizer.fit(X_std)       
visualizer.show() 


# ## silhouette_score

# In[32]:


from sklearn.metrics import silhouette_score

range_n_clusters = list(range(2, 11))
print(range_n_clusters)


# In[33]:


arr_silhouette_score_euclidean = []
for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i).fit(X_std)
    preds = kmeans.predict(new_df)
     
    score_euclidean = silhouette_score(new_df, preds, metric='euclidean')
    arr_silhouette_score_euclidean.append(score_euclidean)


# In[34]:


fig, ax = plt.subplots(1,2,figsize=(15, 6))
sns.lineplot(x=range(2, 11), y=arr_silhouette_score_euclidean, color='#000087', linewidth = 4, ax=ax[0])
sns.scatterplot(x=range(2, 11), y=arr_silhouette_score_euclidean, s=300, color='#800000',  linestyle='--',ax=ax[0])

sns.lineplot(x=range(2, 11), y=inertia, color='#000087', linewidth = 4,ax=ax[1])
sns.scatterplot(x=range(2, 11), y=inertia, s=300, color='#800000',  linestyle='--', ax=ax[1])


# In[ ]:





# In[ ]:





# In[ ]:





# In[35]:


from sklearn.metrics import silhouette_score

# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(X_std)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(X_std, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
    


# In[36]:


# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
silhouette_scores = []  # Initialize an empty list to store the silhouette scores

for num_clusters in range_n_clusters:
    # Initialize kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(X_std)
    
    cluster_labels = kmeans.labels_
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_std, cluster_labels)
    silhouette_scores.append(silhouette_avg)  # Append the silhouette score to the list
    
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))

# Create a line plot of silhouette scores
plt.plot(range_n_clusters, silhouette_scores, marker='o')
plt.title('Silhouette Analysis')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# In[37]:


from yellowbrick.cluster import KElbowVisualizer

model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,11), metric='silhouette', timings=True, locate_elbow=True)
visualizer.fit(X_std)        
visualizer.show()    


# In[38]:


# silhouette plot
from yellowbrick.cluster import SilhouetteVisualizer

for i in [2,3,4,5]:
    model = KMeans(i, random_state=123)
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(X_std)        
    visualizer.show()


# In[ ]:





# In[ ]:





# In[39]:


from sklearn.cluster import KMeans  # Import the KMeans class

kmeans = KMeans(n_clusters=4, random_state=0)  # Create a KMeans instance with 3 clusters
kmeans.fit(X_std)  # Fit the KMeans model to the standardized feature values

new_data['cluster'] = kmeans.labels_
new_data.head(10)


# In[40]:


fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(data=new_data, x='Qty', y='TotalAmount', hue='cluster')


# In[41]:


display(new_data.groupby('cluster').agg(['mean','median']))


# In[42]:


display(new_data.groupby('cluster').agg(['min','max']))


# ## Create new clustering model with 3 cluster

# In[43]:


from sklearn.cluster import KMeans  # Import the KMeans class

kmeans = KMeans(n_clusters=3, random_state=0)  # Create a KMeans instance with 3 clusters
kmeans.fit(X_std)  # Fit the KMeans model to the standardized feature values

new_data['cluster'] = kmeans.labels_
new_data.head(10)


# In[44]:


fig, ax = plt.subplots(figsize=(7,5))
sns.scatterplot(data=new_data, x='Qty', y='TotalAmount', hue='cluster')


# In[45]:


display(new_data.groupby('cluster').agg(['mean','median']))


# In[ ]:




