#!/usr/bin/env python
# coding: utf-8

# In[27]:


# data
import json
import pandas as pd
import numpy as np


# In[97]:


# visualization
from bokeh.models.sources import ColumnDataSource, CDSView
from bokeh.layouts import column, gridplot, row
from bokeh.transform import transform, Jitter, linear_cmap
from bokeh.models import (
                        DataTable, TableColumn, Panel, Tabs,
                        Button, Slider, MultiChoice,Dropdown,RadioButtonGroup,
                        ColorBar, LinearColorMapper,
                        )

from bokeh.palettes import RdYlBu5, Category10, Turbo256, Inferno256
from bokeh.plotting import figure, curdoc, show
from jinja2 import Template

# machine learning
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

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


# In[98]:


# ==== util functions ====
def get_second(x):
  return x[1]
def get_rank(data):
    dataID = [(i,x) for i, x in enumerate(data)]
    dataID.sort(key=get_second)
    ranks = [0]* len(data)
    for i in range(len(ranks)):
        ranks[dataID[i][0]] = i
    return ranks


# In[99]:


# load data
df = pd.read_csv('CLUSTER.csv')


# In[100]:


df


# In[180]:


# get base stats on each column/varaible of our dataset
df_description = df.describe().T


# In[181]:


df_description


# In[184]:


# transform to bokeh data format 
tableSource = ColumnDataSource(df_description)


# In[186]:


# create a interactive table
tableColumns = []
for colName in tableSource.data.keys():
    tableColumns.append(TableColumn(field=colName, title=colName))
data_table = DataTable(
    source=tableSource,
    columns=tableColumns, 
    width=600, height=900,
    sizing_mode="stretch_both"
)


# In[187]:


show(data_table)


# In[105]:


distData["rank"] = get_rank(df["STREETS-TYPE"])


# In[106]:


distData = pd.DataFrame()


# In[107]:


distData


# In[170]:


output_file("layout.html")


# In[196]:


# ==== creating figures and plotting data  =====

# create ColumnDataSource for rank/value + map plot
distData = pd.DataFrame()
distData["value"]= df["TREE-CANOPY AREA"]
distData["rank"] = get_rank(df["TREE-CANOPY AREA"])
distData["x_coord"] =  df["X"]
distData["y_coord"] = df["Y"]


# In[197]:


# transform to bokeh data format
distSource = ColumnDataSource(distData)


# In[244]:


#create figure for rank/value plot
distFig = figure(
            plot_width=300,
            plot_height=200,
            toolbar_location ="left",
            tools="lasso_select"
            )


# In[245]:


# add glyphs to the figure 
distFig.circle(
    x="rank",
    y="value",
    fill_color = "cyan",
    selection_fill_color = "red",
    selection_line_width = 2,
    selection_line_color = "red",
    fill_alpha = 0.05,
    line_width=0,
    muted_alpha = 0.05,
    size=4,
    source=distSource,
    name = "distFig_circle"
)


# In[251]:


#create figure for map plot
mapFig = figure(
        plot_width=300,
        plot_height=200,
        toolbar_location ="left",
        tools="lasso_select"
    )


# In[277]:


# add glyphs to map plot

mapper= LinearColorMapper(palette=Turbo256)
mapFig.circle(
    x="X",
    y="Y",
    fill_color = {'field': 'STREETS-2019-25FEB', 'transform': mapper},
    selection_fill_color = "red",
    fill_alpha = 1,
    line_width=0,
    muted_alpha = 0.4,
    size=4,
    source=distSource,
    name="mapFig_circle"
)


# In[278]:


# == callbacks: add interactivity ==
def updateDistCDS(selectionName):
# re-create ColumnDataSource for dist & map figure
    distData = pd.DataFrame()
    distData["value"]= df["selectionName"]
    distData["rank"] = get_rank(df["selectionName"])
    distData["x_coord"] =  df["X"]
    distData["y_coord"] = df["Y"]


# In[279]:


# update data
distFig.select("distFig_circle").data_source.data = distData
mapFig.select("mapFig_circle").data_source.data = distData


# In[306]:


# trigger a callback when a row on the table is selected 
def tableSelection_callback(attrname, old, new):
    # get row id
    selectionIndex=tableSource.selected.indices[0]
    # translate to column name
    selectionName= tableSource.data["index"][selectionIndex]
    # call functio to update plots
    updateDistCDS(selectionName)

tableSource.selected.on_change('indices', tableSelection_callback)


# In[308]:


# update titles
distFig.title = selectionName + " distribution"
mapFig.title = selectionName + " map"


# In[312]:


# ===================================================================
# Part II: run regression models, visualize and compare their results
# ===================================================================

# widgets for model parameter 
slider_Cluster = Slider(start=1, 
                          end=10, 
                          value=1, 
                          step=1, 
                          title="No.s of Clusters", 
                          name="slider_Cluster")

cluster_Mode = RadioButtonGroup(labels=["Agglomerative", "K-Means"], 
                                    active=0,
                                    name="cluster_Mode")


# In[313]:


show(cluster_Mode)


# In[284]:


compute_model= Button(label="compute clusters first",width =100, name="compute_model")


# In[285]:


show(compute_model)


# In[321]:


# ================================================================
# translate data-prep and regression code into a callback/function

def computeAgglomerative(ts_slider, 
                        tsMode, 
                        dataCDS, 
                        df, 
                        OUTPUT_COLUMN='STREETS-TYPE'):
    
    mbi_df = df.copy()
    
    # read-out user settings from UI widgets 
    tsMode = cluster_Mode.active
    if tsMode == 0:
        # extract id's from selected datapoints from a ColumnDataSource
        testIDs = dataCDS.selected.indices
        test_df = mbi_df.iloc[testIDs]
    elif tsMode==1:
        # number of rows in a CDS
        test_df = mbi_df.sample(frac=ts_slider.value,random_state=0)

    train_df = mbi_df.drop(test_df.index)

    
    # Seperate input and output columns
    train_input_df = train_df.copy()
    train_output_df = train_input_df.pop(OUTPUT_COLUMN)

    test_input_df = test_df.copy()
    test_output_df = test_input_df.pop(OUTPUT_COLUMN)

    # Scalers
    input_scaler = MinMaxScaler()
    scaled_train_input_df = pd.DataFrame(
        input_scaler.fit_transform(train_input_df),
        columns=train_input_df.columns
    )
    scaled_test_input_df = input_scaler.transform(test_input_df)

    output_scaler = MinMaxScaler()
    scaled_train_output_df = pd.DataFrame(
        output_scaler.fit_transform(np.array(train_output_df).reshape(-1, 1)),
        columns=[OUTPUT_COLUMN]
    )
    scaled_test_output_df = output_scaler.transform(np.array(test_output_df).reshape(-1, 1))


# # Agglomerative Clustering

# In[315]:


# Input shape
#input_shape = [*scaled_train_input_df.shape[1:]]

   # Extract coords from data
df_coords = df[['X', 'Y']].copy()
df_data   = df.drop(['X', 'Y'], axis=1)

# Scale data
scaler = MinMaxScaler()
scaled_df = pd.DataFrame(
scaler.fit_transform(df_data),
columns=df_data.columns
)


# In[316]:


from sklearn.cluster import AgglomerativeClustering, KMeans


# In[338]:


#Build and compile models
ac_model = AgglomerativeClustering(
    n_clusters=10,
    # n_clusters=None,
    # distance_threshold=0
)


# In[339]:


# Train models
ac_model.fit(scaled_df)


# In[340]:


# Get classes
ac_model.labels_


# In[341]:


def map_clustering_results(coords, model):
    labels = model.labels_
    for label in np.unique(labels):
        x = coords['X'].to_numpy()[labels == label]
        y = coords['Y'].to_numpy()[labels == label]
        plt.scatter(x, y, label=model.labels_)


# In[342]:


map_clustering_results(df_coords, ac_model)


# In[295]:


# re-create CDS with new data and update
newData={}
newData["y_log"] = log_ac_history.history['loss']
newData["y_NN"] = neural_net_history.history['loss']
newData["x_log"] = range(len(log_reg_history.history['loss']))
newData["x_NN"] = range(len(neural_net_history.history['loss']))
lossCDS.data = newData


# In[322]:


# set-up callback
def cb_compute_model(event):
    computeAgglomerative(slider_Cluster,
                    cluster_Mode, 
                    distSource,
                    df=df, 
                    OUTPUT_COLUMN = 'STREETS-TYPE')

# assign callback to UI widget
compute_model.on_click(cb_compute_model)


# In[327]:


# create layout

bokehLayout = row(data_table,
                    column(
                        row(distFig,mapFig),
                        row(slider_TestSplit,rb_TestSplitMode,compute_model),
                        lossFig),
                    name="bokehLayout")


# In[326]:


# add to curDoc
curdoc().add_root(bokehLayout)


# In[ ]:





# In[ ]:




