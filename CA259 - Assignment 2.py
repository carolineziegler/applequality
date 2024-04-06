#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


inpPath = "C:/CarolineZiegler/Studium_DCU/8. Semester/Data Analytics for Marketing Applications/Pair Assignments/Assignment 2/"
apples = pd.read_csv(inpPath + "apple_quality.csv", delimiter =  ",", header = 0)
apples


# In[3]:


apples = apples.drop(4000)
apples


# In[4]:


apples.drop("A_id", axis=1, inplace = True)
apples


# In[5]:


apples.describe().style.background_gradient(axis=1, cmap='Blues')


# In[6]:


type(apples.iloc[0,6])


# In[7]:


apples["Acidity"] = apples["Acidity"].astype("float")
type(apples.iloc[0,6])


# In[74]:


apples.describe().style.background_gradient(axis=1, cmap='Blues')


# In[9]:


apples.isna().sum()


# In[10]:


sns.histplot(apples["Size"])
plt.show()


# In[11]:


print(apples["Size"].skew())
print(apples["Size"].kurtosis())


# In[12]:


sns.histplot(apples["Weight"])
plt.show()


# In[13]:


print(apples["Weight"].skew())
print(apples["Weight"].kurtosis())


# In[14]:


sns.histplot(apples["Sweetness"])
plt.show()


# In[15]:


print(apples["Sweetness"].skew())
print(apples["Sweetness"].kurtosis())


# In[16]:


sns.histplot(apples["Crunchiness"])
plt.show()


# In[17]:


print(apples["Crunchiness"].skew())
print(apples["Crunchiness"].kurtosis())


# In[18]:


sns.histplot(apples["Juiciness"])
plt.show()


# In[19]:


print(apples["Juiciness"].skew())
print(apples["Juiciness"].kurtosis())


# In[20]:


sns.histplot(apples["Ripeness"])
plt.show()


# In[21]:


print(apples["Ripeness"].skew())
print(apples["Ripeness"].kurtosis())


# In[22]:


sns.histplot(apples["Acidity"])
plt.show()


# In[23]:


print(apples["Acidity"].skew())
print(apples["Acidity"].kurtosis())


# In[24]:


#gettig an overview over all histograms understanding normal distribution
numerical_columns = ['Size', 'Weight', 'Sweetness', 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity']

plt.figure(figsize=(15, 10))
sns.set_palette("tab10")


for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=apples, x=column, kde=True, bins=20)  
    plt.title(column)

plt.tight_layout()
plt.show()


# In[25]:


apples["Quality"].unique()


# In[26]:


quality_counts = apples["Quality"].value_counts()

plt.pie(quality_counts, labels=quality_counts.index, autopct='%1.1f%%', textprops={'fontsize':10})
plt.title('Quality Distribution')
plt.show()


# In[27]:


encoded_dict = {'good': 1, 'bad': 0}
apples["Quality_numeric"] = apples["Quality"].map(encoded_dict)
apples


# In[28]:


plt.figure(figsize=(15, 10))
sns.set_palette("Set2")

for i, column in enumerate(apples.columns[:-2]):  
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x='Quality_numeric', y=column, data=apples)
    plt.title(f'{column} by Quality_numeric')

plt.tight_layout()
plt.show()


# In[29]:


plt.figure(figsize=(8, 8))
sns.set(style="white")  

sns.jointplot(x='Size', y='Weight', hue='Quality_numeric', data=apples, palette='tab10', s=9)


# In[30]:


apples_corr = apples.corr()
apples_corr


# In[31]:


sns.heatmap(apples_corr, cmap= "Blues", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Heatmap')
plt.show()


# In[32]:


mask = np.triu(np.ones_like(apples_corr, dtype=bool))

plt.figure(figsize=(10, 8))

sns.heatmap(apples_corr, mask=mask, cmap= "Blues", vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Heatmap')
plt.show()


# In[33]:


#Analysing the correlations pairs with a correlation higher or lower and equal to 0.2 and -0.2
tpllst = []
for i in range(0,8):
    for j in range(0,8):
        if apples_corr.iloc[i,j]>=0.2 and apples_corr.iloc[i,j]<1: 
            tpllst.append((apples_corr.index[i], apples_corr.index[j], apples_corr.iloc[i,j]))
        elif apples_corr.iloc[i,j]<= -0.2 and apples_corr.iloc[i,j]>-1:
            tpllst.append((apples_corr.index[i], apples_corr.index[j], apples_corr.iloc[i,j]))

dltlst = [] #new list to save only those items where the quality_numeric is not at the second place underlining that this variable makes no sense as independent variable
for i in range(len(tpllst)):
    if tpllst[i][1] != "Quality_numeric":
        dltlst.append(tpllst[i])
dltlst


# In[34]:


#linear regressions for variables with 2 or more correlations above/equal 0.2 or below/equal -0.2

# import libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


# In[35]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = apples[['Size','Ripeness']]
yDf = apples['Sweetness']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_sweet = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_sweet.score(X_test, y_test))
print(reg_lin_sweet.coef_)


# In[36]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = apples[['Juiciness','Ripeness']]
yDf = apples['Crunchiness']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_crunch = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_crunch.score(X_test, y_test))
print(reg_lin_crunch.coef_)


# In[37]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = apples[['Weight','Sweetness', 'Crunchiness', 'Acidity']]
yDf = apples['Ripeness']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_ripe = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_ripe.score(X_test, y_test))
print(reg_lin_ripe.coef_)


# In[38]:


# Split the data columnwise into the independt variables (x) and the dependent variable (y)
xDf = apples[['Juiciness','Ripeness']]
yDf = apples['Acidity']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

#fitting the model
reg_lin_ripe = LinearRegression().fit(X_train, y_train)

#getting the accuracy
print(reg_lin_ripe.score(X_test, y_test))
print(reg_lin_ripe.coef_)


# In[39]:


from sklearn.linear_model import LogisticRegression
xDf = apples[['Size','Sweetness','Juiciness','Ripeness']]
yDf = apples['Quality_numeric']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(X_train, y_train)


# Evaluate the accuracy of the model
print(logreg.score(X_test, y_test))


# In[40]:


# Access the coefficient values
coefficients = logreg.coef_

# Calculate the odds ratios
odds_ratios = np.exp(coefficients)

print("Shape of odds_ratios:", odds_ratios.shape)
print("Number of columns in apples[['Size','Sweetness','Juiciness','Ripeness']]:", len(apples[['Size','Sweetness','Juiciness','Ripeness']].columns))

# Iterate over the independent variables and their odds ratios
for i, feature in enumerate(apples[['Size','Sweetness','Juiciness','Ripeness']].columns):
    print(f"{feature}: Odds Ratio = {odds_ratios[0, i]}")


# In[41]:


xDf = apples[['Size']]
yDf = apples['Quality_numeric']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

logreg = LogisticRegression()

# Fit the model to the training data
logreg.fit(X_train, y_train)


# Evaluate the accuracy of the model
print(logreg.score(X_test, y_test))


# In[42]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x=apples['Size'], y=apples['Quality_numeric'], color='blue', label='Data Points')

# Plot logistic function
x_values = np.linspace(xDf.min(), xDf.max(), 100).reshape(-1, 1)
y_proba = logreg.predict_proba(x_values)[:, 1]  # Probability of class 1
plt.plot(x_values, y_proba, color='red', label='Logistic Function')

plt.xlabel('Feature')
plt.ylabel('Probability')
plt.title('Logistic Regression')
plt.legend()
plt.show()


# In[43]:


import random

random.seed(42)

#optional
xDf = apples.drop(columns='Quality')
#correct import
from sklearn.cluster import KMeans

#initialize inertia list and get the k and the inertia per k (or WCSS per k)
inertiaLst = []
for kVal in range(1, 10):
    kmeans = KMeans(n_clusters=kVal)
    kmeans.fit(xDf)
    inertiaLst.append([kVal, kmeans.inertia_])

#transposing and plotting the inertia list
inertiaArr = np.array(inertiaLst).transpose()
plt.plot(inertiaArr[0], inertiaArr[1])
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()


# In[44]:


#read kVal at the "elbow"
kVal = 3

#set the k for the KMeans to kVal
kmeans = KMeans(n_clusters=kVal)

#fit the model
kmeans.fit(xDf)

#get the label from the kmeans and add it as a column to the df
apples['label'] = kmeans.labels_


# In[45]:


apples


# In[46]:


line_colors = ['blue','green', 'gray'] 
label_counts = apples["label"].value_counts()

plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', textprops={'fontsize':10}, colors = line_colors)
plt.title('Label Distribution')
plt.show()


# In[47]:


label_counts


# In[48]:


apples.drop(columns = "Quality").groupby(by = "label").mean()


# In[49]:


centroids = kmeans.cluster_centers_
print(centroids)


# In[51]:


centroids = kmeans.cluster_centers_
feature_names = ['Size', 'Weight', 'Sweetness' , 'Crunchiness', 'Juiciness', 'Ripeness', 'Acidity' ,'Quality_numeric']

# Create a DataFrame to store the centroids and feature names
centroid_table = pd.DataFrame(centroids, columns=feature_names)

centroid_table


# In[53]:


transposed_centroid = centroid_table.transpose()
transposed_centroid


# In[54]:


line_colors = ['gray', 'green', 'blue'] 
fig = plt.figure(figsize=(12, 8))
for i, column in enumerate(transposed_centroid.columns):
    plt.plot(transposed_centroid[column], marker='o', linestyle='', markersize=12, color=line_colors[i])
plt.title('Characteristics for each Apple Variety', fontweight='bold')
plt.xlabel("Features", fontweight='bold')
plt.ylabel("Centroid Value", fontweight='bold')
plt.legend(transposed_centroid.columns)


# In[55]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Sweetness', y='Juiciness', hue='label', data=apples, palette=line_colors, s=9)


# In[56]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Sweetness', y='Ripeness', hue='label', data=apples, palette=line_colors, s=9)


# In[57]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Sweetness', y='Acidity', hue='label', data=apples, palette=line_colors, s=9)


# In[58]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Juiciness', y='Ripeness', hue='label', data=apples, palette=line_colors, s=9)


# In[59]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Juiciness', y='Acidity', hue='label', data=apples, palette=line_colors, s=9)


# In[60]:


plt.figure(figsize=(15, 10))
sns.set(style="white") 

sns.jointplot(x='Ripeness', y='Acidity', hue='label', data=apples, palette=line_colors, s=9)


# In[61]:


apples_0 = apples[apples["label"] == 0]
apples_0


# In[62]:


apples_1 = apples[apples["label"] == 1]
apples_1


# In[63]:


apples_2 = apples[apples["label"] == 2]
apples_2


# In[64]:


from scipy import stats


# In[79]:


# Sample data
sweet_1 = apples_1["Sweetness"]
sweet_2 = apples_2["Sweetness"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(sweet_2, sweet_1)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[78]:


# Sample data
juicy_1 = apples_1["Juiciness"]
juicy_0 = apples_0["Juiciness"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(juicy_0, juicy_1)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[77]:


# Sample data
ripe_1 = apples_1["Ripeness"]
ripe_0 = apples_0["Ripeness"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(ripe_1, ripe_0)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[75]:


# Sample data
acid_2 = apples_2["Acidity"]
acid_0 = apples_0["Acidity"]

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(acid_0, acid_2)

# Print the results
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[80]:


#Besides just including 4 independent variables, all features were included leading to a even higher significance
from sklearn.linear_model import LogisticRegression
xDf = apples[['Size','Sweetness','Juiciness','Ripeness', 'Weight', 'Acidity', 'Crunchiness']]
yDf = apples['Quality_numeric']

# Split between train and test set
X_train, X_test, y_train, y_test = train_test_split(xDf, yDf, test_size=0.3, random_state=0)

logreg2 = LogisticRegression()

# Fit the model to the training data
logreg2.fit(X_train, y_train)


# Evaluate the accuracy of the model
print(logreg2.score(X_test, y_test))
print(logreg2.coef_)


# In[81]:


#instead of manually entering the clusters max and min values another code was created which is replicable for other projects as well
sweet_max = centroid_table['Sweetness'].idxmax()
sweet_min = centroid_table['Sweetness'].idxmin()

if sweet_max == 0:
    sweet_max_df = apples_0['Sweetness']
elif sweet_max == 1:
    sweet_max_df = apples_1['Sweetness']
elif sweet_max == 2:
    sweet_max_df = apples_2['Sweetness']
    

if sweet_min == 0:
    sweet_min_df = apples_0['Sweetness']
elif sweet_min == 1:
    sweet_min_df = apples_1['Sweetness']
elif sweet_min == 2:
    sweet_min_df = apples_2['Sweetness']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(sweet_max_df, sweet_min_df)

# Print the results
print(f'Test: Is Apple Variaty {sweet_max} systematically more sweet than Apple Variaty {sweet_min}?' )
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[82]:


juicy_max = centroid_table['Juiciness'].idxmax()
juicy_min = centroid_table['Juiciness'].idxmin()

if juicy_max == 0:
    juicy_max_df = apples_0['Juiciness']
elif juicy_max == 1:
    juicy_max_df = apples_1['Juiciness']
elif juicy_max == 2:
    juicy_max_df = apples_2['Juiciness']
    

if juicy_min == 0:
    juicy_min_df = apples_0['Juiciness']
elif juicy_min == 1:
    juicy_min_df = apples_1['Juiciness']
elif juicy_min == 2:
    juicy_min_df = apples_2['Juiciness']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(juicy_max_df, juicy_min_df)

# Print the results
print(f'Test: Is Apple Variaty {juicy_max} systematically more juicy than Apple Variaty {juicy_min}?' )
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[83]:


ripe_max = centroid_table['Ripeness'].idxmax()
ripe_min = centroid_table['Ripeness'].idxmin()

if ripe_max == 0:
    ripe_max_df = apples_0['Ripeness']
elif ripe_max == 1:
    ripe_max_df = apples_1['Ripeness']
elif ripe_max == 2:
    ripe_max_df = apples_2['Ripeness']
    

if ripe_min == 0:
    ripe_min_df = apples_0['Ripeness']
elif ripe_min == 1:
    ripe_min_df = apples_1['Ripeness']
elif ripe_min == 2:
    ripe_min_df = apples_2['Ripeness']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(ripe_max_df, ripe_min_df)

# Print the results
print(f'Test: Is Apple Variaty {ripe_max} systematically more ripe than Apple Variaty {ripe_min}?' )
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[84]:


acid_max = centroid_table['Acidity'].idxmax()
acid_min = centroid_table['Acidity'].idxmin()

if acid_max == 0:
    acid_max_df = apples_0['Acidity']
elif acid_max == 1:
    acid_max_df = apples_1['Acidity']
elif acid_max == 2:
    acid_max_df = apples_2['Acidity']
    

if acid_min == 0:
    acid_min_df = apples_0['Acidity']
elif acid_min == 1:
    acid_min_df = apples_1['Acidity']
elif acid_min == 2:
    acid_min_df = apples_2['Acidity']

# Perform independent t-test
t_statistic, p_value = stats.ttest_ind(acid_max_df, acid_min_df)

# Print the results
print(f'Test: Have Apples from Variaty {acid_max} systematically more Acidity than Apples from Variaty {acid_min}?' )
print("t-statistic:", t_statistic)
print("p-value:", p_value)


# In[ ]:




