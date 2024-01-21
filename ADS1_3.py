# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 16:48:28 2024

@author: DELL
"""

import pandas as pd
import numpy as np

import sklearn.cluster as cluster
import sklearn.metrics as skmet
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cluster_tools as ct

help(ct)

# Load the data
data = pd.read_csv("world_Bank_Data.csv")

# Explore the data
print(data.head())



# Normalize relevant indicators
data['normalized_gdp'] = data['CO2 emissions (kg per PPP $ of GDP)'] / data['totalpopulation']
data['normalized_co2_per_capita'] = data['CO2 emissions (kt)'] / data['totalpopulation']
# Add more normalization as needed
# extract columns for fitting. 
# .copy() prevents changes in df_fit to affect df_fish. 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        

df_fit = data[['normalized_gdp', 'normalized_co2_per_capita']].copy()
# Extract features for clustering
features = data[['normalized_gdp', 'normalized_co2_per_capita']]

# Drop missing values from the DataFrame used for clustering
df_fit = df_fit.dropna()

# Apply K-Means clustering
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    data['cluster'] = kmeans.fit_predict(df_fit)
    
    
 # Plot cluster membership
plt.scatter(data['normalized_gdp'], data['normalized_co2_per_capita'], c=data['cluster'], cmap='viridis')
plt.title('Cluster Membership')
plt.xlabel('Normalized GDP')
plt.ylabel('Normalized CO2 per Capita')
plt.show()

# Define a simple model function (e.g., exponential growth)

"""
    Exponential growth model function.

    Parameters:
    - x: Input variable (e.g., year)
    - a, b: Parameters to be optimized

    Returns:
    - Exponential growth model prediction
    """
def model_func(x, a, b):
    return a * np.exp(b * x)

# Fit the model to the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_gdp = scaler.fit_transform(data['normalized_gdp'].values.reshape(-1, 1)).flatten()

popt, pcov = curve_fit(model_func, data['year'], data['normalized_gdp'], maxfev=2000, p0=[2.0, 0.01])
from lmfit import Model

model = Model(model_func)
result = model.fit(data['normalized_gdp'], x=data['year'], a=2.0, b=0.01)

# Get parameter values
popt = result.params.valuesdict()



# Generate predictions for the next 10 years
future_years = np.arange(data['year'].max() + 1, data['year'].max() + 11)
predicted_values = model_func(future_years, *popt)

# Plot the best-fitting function and confidence range
plt.scatter(data['year'], data['normalized_gdp'], label='Data')
plt.plot(future_years, predicted_values, label='Best-Fitting Function', color='red')
plt.fill_between(future_years, predicted_values - pcov[0, 0], predicted_values + pcov[0, 0], color='pink', alpha=0.5, label='Confidence Range')
plt.title('Best-Fitting Function with Confidence Range')
plt.xlabel('Year')
plt.ylabel('Normalized GDP')
plt.legend()
plt.show()