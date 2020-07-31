# Udacity-Machine-Learning-Engineer-Nanodegree-Capstone-Project

Source code list:
1. Arvato Kmeans: The code used to generate the Arvato Customer Segmentation Analysis report
2. Arvato SVM: The code used for creating the SVM model to do the customer response predicting model.
3. helper: the helper funtion created for data cleaning process

Attention: The Arvato Kmeans.py was run on my local computer, To run the code successfully on your computer,
you need to create a folder named "Kmeans_raw_data" in your code folder and store the downloarded Arvato raw data
into this folder. Udacity_AZDIAS_052018.csv will be named as azdias.csv; Udacity_CUSTOMERS_052018.csv will be 
renamed as customers.csv


Package List:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import helper
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

ps: To import imblearn, you need to 'pip install imbalanced-learn' first in the workspace terminal
import imblearn

from imblearn.over_sampling import SMOTE<br>
from imblearn.under_sampling import RandomUnderSampler<br>
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import svm
import os
from sklearn import metrics
from pandas import Series
