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
import numpy as np<br>
import pandas as pd<br>
import matplotlib.pyplot as plt<br>
import seaborn as sns<br>
from sklearn.decomposition import PCA<br>
from sklearn.cluster import KMeans<br>
from sklearn.metrics import silhouette_score<br>
import helper<br>
from sklearn.preprocessing import MinMaxScaler<br>
from collections import Counter<br>

ps: To import imblearn, you need to 'pip install imbalanced-learn' first in the workspace terminal
import imblearn<br>
from imblearn.over_sampling import SMOTE<br>
from imblearn.under_sampling import RandomUnderSampler<br>
from imblearn.pipeline import Pipeline<br>
from sklearn.model_selection import train_test_split<br>
from sklearn import svm<br>
import os<br>
from sklearn import metrics<br>
from pandas import Series<br>
