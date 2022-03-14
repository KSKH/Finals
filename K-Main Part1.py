#!/usr/bin/env python
# coding: utf-8

# In[53]:


# Data manipulation
import json 
import numpy as np
import pandas as pd
import seaborn as sns

# Visualisation
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from bokeh.io import output_file, show, output_notebook
from bokeh.models import GeoJSONDataSource,LinearColorMapper,HoverTool,CustomJS,Dropdown,RadioButtonGroup
from bokeh.plotting import figure
from bokeh.palettes import Viridis6,Turbo256, Category20
from bokeh.layouts import column, row

# Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# In[54]:


# Upload CSV file and store in a Pandas DataFrame
df = pd.read_csv('MASTERDATA_1.csv')


# In[55]:


df


# In[56]:


df.describe().T


# In[57]:


df


# In[58]:


df.dtypes


# In[64]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
ohe=OneHotEncoder()
print(ohe)


# In[65]:


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

df2=MultiColumnLabelEncoder(columns = ['STREETS-TYPE','TREE-CONDITION','TREE-OWNERSHIP','TREE-VALUE']).fit_transform(df)


# In[66]:


df2


# In[67]:


df3


# In[68]:


# Visualisation
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from bokeh.io import output_file, show, output_notebook
from bokeh.models import GeoJSONDataSource,LinearColorMapper,HoverTool,CustomJS,Dropdown,RadioButtonGroup
from bokeh.plotting import figure
from bokeh.palettes import Viridis6,Turbo256, Category20
from bokeh.layouts import column, row

# Machine Learning
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model


# In[69]:


df2.describe().T


# In[70]:


# Finding correlation
df3 = df2.drop(['id','X','Y'],axis=1)
corr = df3.corr()
c1 = corr.abs().unstack()
list(c1.sort_values(ascending=False).items())[len(df.columns)::2][:8]


# In[71]:


df3=df2.replace(np.nan, 0)


# In[72]:


df3


# In[31]:


df3.to_csv('C:/Users/ksham/Desktop/AI IN URBANISM/THESIS/CLIPPED AREA 4/CLUSTER.csv', index=False)


# In[73]:


# Extract coords from data# Extract coords from data
df_coords = df3[['X','Y']].copy()
df_data   = df3.drop(['X', 'Y','id','fid'], axis=1)


# In[74]:


# Scale data
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(
    scaler.fit_transform(df_data),
    columns=df_data.columns
)


# In[75]:


scaled_df


# # Agglomerative Clustering

# In[76]:


# Build and compile models
ac_model = AgglomerativeClustering(
    n_clusters=10,
    # n_clusters=None,
    # distance_threshold=0
)


# In[77]:


# Train models
ac_model.fit(scaled_df)


# In[78]:


# Get classes
ac_model.labels_


# In[79]:


def map_clustering_results(coords, model):
    labels = model.labels_
    for label in np.unique(labels):
        x = coords['X'].to_numpy()[labels == label]
        y = coords['Y'].to_numpy()[labels == label]
        plt.scatter(x, y, label=model.labels_)


# In[80]:


map_clustering_results(df_coords,ac_model)


# # K-Means Clustering

# In[81]:


# Build and compile model
km_model = KMeans(
    n_clusters=10
)


# In[82]:


km_model


# In[83]:


km_model.fit(scaled_df)


# In[84]:


def map_clustering_results(coords, model):
    labels = model.labels_
    for label in np.unique(labels):
        x = coords['X'].to_numpy()[labels == label]
        y = coords['Y'].to_numpy()[labels == label]
        plt.scatter(x, y, label=model.labels_)


# In[85]:


map_clustering_results(df_coords,km_model)


# In[86]:


df_cluster = pd.DataFrame(km_model.labels_)


# In[87]:


df_cluster


# In[88]:


df_cluster.describe().T


# In[89]:


df_cluster0 = df.loc[df_cluster[0]==0]
df_cluster1 = df.loc[df_cluster[0]==1]
df_cluster2 = df.loc[df_cluster[0]==2]
df_cluster3 = df.loc[df_cluster[0]==3]
df_cluster4 = df.loc[df_cluster[0]==4]
df_cluster5 = df.loc[df_cluster[0]==5]
df_cluster5 = df.loc[df_cluster[0]==6]
df_cluster5 = df.loc[df_cluster[0]==7]
df_cluster5 = df.loc[df_cluster[0]==8]
df_cluster5 = df.loc[df_cluster[0]==9]


# In[90]:


df_cluster0.describe().T


# # Autoencoders & K-Means Clustering

# In[91]:


# Collect clustering results
additional_attributes = {
    "Agglomerative": ac_model.labels_,
    "K-Means": km_model.labels_
}


# In[52]:


# Define output file for bokeh
output_notebook()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




