#!/usr/bin/env python
# coding: utf-8

from IPython.display import display, get_ipython

# In[ ]:


get_ipython().run_line_magic('pip', 'install kagglehub')


# In[ ]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("ahmedwaelnasef/cars-dataset")
get_ipython().system('mkdir data')
get_ipython().system('cp {path}/cars.csv data/cars.csv')


# In[126]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


# In[3]:


df = pd.read_csv('data/cars.csv')
df.head()
df['seats'].value_counts()  


# In[ ]:





# In[ ]:


df[df['top_speed'] == 'Automatic']


# In[ ]:


df[df['car name'] == 'Mercedes-Benz EQV 2021 6 seater']


# In[4]:


# Replace non-numeric values in horse_power with NaN
df['horse_power'] = df['horse_power'].replace(r'^\D*$', np.nan, regex=True)
df.isna().sum()


# In[6]:


df.dropna(subset=['horse_power'], inplace=True)
df.isna().sum()


# In[14]:


mask = df['top_speed'].str.contains(r'^\D*$', regex=True)
df.drop(df[mask].index, inplace=True)
df.isna().sum()
df.reset_index(drop=True, inplace=True)


# In[15]:


df.info()


# In[19]:


df[df['cylinder'] == 'N/A, Electric']


# In[20]:


string_columns = df.select_dtypes(include=['object']).columns
df[string_columns] = df[string_columns].apply(lambda x: x.str.strip())


# In[21]:


df.shape[0] - df[df['price'].str.contains(r'\d', regex=True)].shape[0]


# In[22]:


dataframe_no_price = df[df['price'].str.contains(r'^\D*$', regex=True) == True]['price']
dataframe_no_price.unique()


# In[ ]:


dataframe_no_price.nunique()


# In[23]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.info()


# In[ ]:


df['price'] = df['price'].astype(float)


# In[ ]:


df.isna().sum()


# In[24]:


# Extract currency and number
pattern = r'([A-Z]+)\s([\d,]+)'
extracted = df['price'].str.extract(pattern, expand=True)

# Rename columns for clarity
extracted.columns = ['currency', 'amount']

df = pd.concat([df, extracted], axis=1)
df.drop(columns='price', inplace=True)

df.rename(columns={'amount': 'price'}, inplace=True)

# # Handle non-matching rows (e.g., fill NaN with 'Unknown' for currency and 0 for amount)
# extracted['currency'].fillna('Unknown', inplace=True)
# extracted['amount'].fillna('0', inplace=True)

# print(extracted)
df.head()


# In[25]:


df['price'] = df['price'].str.replace(',', '').astype(float)


# In[26]:


# Dictionary for mapping country to currency
country_to_currency = {
    'ksa': 'SAR',      # Saudi Riyal
    'egypt': 'EGP',    # Egyptian Pound
    'bahrain': 'BHD',  # Bahraini Dinar
    'qatar': 'QAR',    # Qatari Riyal
    'oman': 'OMR',     # Omani Rial
    'kuwait': 'KWD',   # Kuwaiti Dinar
    'uae': 'AED'       # United Arab Emirates Dirham
}

# Dictionary for mapping currency to USD exchange rate
currency_to_usd = {
    'SAR': 0.27,  # Saudi Riyal to USD
    'EGP': 0.032, # Egyptian Pound to USD
    'BHD': 2.65,  # Bahraini Dinar to USD
    'QAR': 0.27,  # Qatari Riyal to USD
    'OMR': 2.60,  # Omani Rial to USD
    'KWD': 3.30,  # Kuwaiti Dinar to USD
    'AED': 0.27   # United Arab Emirates Dirham to USD
}

# Map country to currency
df['currency'] = df['country'].map(country_to_currency)

df['price'] = df.apply(lambda row: row['price'] * currency_to_usd[row['currency']], axis=1)

# Compute the mean price for each country
mean_prices = df.groupby('car name')['price'].transform('mean')

# Replace only the missing values with the mean price of their respective groups
df['price'] = df['price'].fillna(mean_prices)


# Convert prices to USD
df.head()


# In[27]:


median_na = df.groupby('brand')['price'].transform('median')
df['price'] = df['price'].fillna(median_na)


# In[ ]:


df.head()


# In[28]:


df.dropna(subset=['price'], inplace=True)
df.isna().sum()


# In[29]:


df.drop(columns='currency', inplace=True)


# In[30]:


df.reset_index(drop=True, inplace=True)


# In[31]:


indices = df[df['top_speed'].str.contains('Seater')].index

temp = df.loc[indices, 'seats']
df.loc[indices, 'seats'] = df['top_speed']
df.loc[indices, 'top_speed'] = temp


# In[32]:


df['top_speed'].str.contains('Seater').sum()


# In[33]:


df.loc[indices].head(50)


# In[34]:


# Columns to process
columns = ['seats', 'engine_capacity', 'brand', 'top_speed', 'cylinder', 'horse_power']

# Function to get the mode (most frequent value)
def get_mode(series):
    mode = series.mode()
    return mode[0] if not mode.empty else None

df[columns] = df.groupby('car name')[columns].transform(get_mode)


# In[101]:


df['top_speed'] = df['top_speed'].astype(int)


# In[102]:


df.info()


# In[37]:


df.isna().sum()


# In[53]:


df['seats'].value_counts()


# will do median imputation based on brand, but first, we have to convert `seats` to an int

# In[62]:


mask = df['seats'].str.contains(r'^\D*$')
df[mask].head(5)


# Convert all values in seats that have no number to `NA`

# In[ ]:


df['seats'] = df['seats'].replace(r'^\D*$', np.nan, regex=True)
df.isna().sum()


# In[65]:


df.head()


# In[61]:


df[df['brand']=='peugeot']


# In[73]:


# Function to extract the number from 'seats' values that contain the string 'Seats'
def extract_seats(value):
    if pd.isna(value):
        return value
    if type(value) == float:
        return int(value)
    if 'Seater' in value or 'Seats' in value:
        return int(value.split()[0])

# Apply the function to the 'seats' column
df['seats'] = df['seats'].apply(extract_seats)


# In[77]:


median = df.groupby('brand')['seats'].transform('median')
df['seats'] = df['seats'].fillna(median)
df.isna().sum()


# In[79]:


df.dropna(subset=['seats'], inplace=True)
df.isna().sum()


# In[81]:


df['seats'] = df['seats'].astype(int)
df.head()


# In[103]:


df['cylinder'] = df['cylinder'].astype(float)
df['horse_power'] = df['horse_power'].astype(int)
df['engine_capacity'] = df['engine_capacity'].astype(float)
df.dtypes


# In[107]:


df[~df['cylinder'].isna()]


# In[153]:


df['cylinder'].value_counts()


# In[110]:


def find_cylinder_count(engine_capacity):
    if engine_capacity < 1.5:
        return 3
    elif engine_capacity < 2:
        return 4
    elif engine_capacity < 2.5:
        return 5
    elif engine_capacity < 4:
        return 6
    elif engine_capacity < 6:
        return 8
    elif engine_capacity < 8:
        return 12
    
df['cylinder'] = df['cylinder'].fillna(df['engine_capacity'].apply(find_cylinder_count))

df.isna().sum()
    
    


# In[ ]:


mask = df[df['engine_capacity'] >= 900]
df.loc[mask.index, 'engine_capacity'] = df.loc[mask.index, 'engine_capacity']/1000
df.loc[mask.index]


# In[150]:


mask = df[(df['engine_capacity'] >= 100) & (df['engine_capacity'] < 500)]
df.loc[mask.index, 'engine_capacity'] = df.loc[mask.index, 'engine_capacity']/100


# In[ ]:





# In[151]:


df['cylinder'] = df['cylinder'].fillna(df['engine_capacity'].apply(find_cylinder_count))
df.isna().sum()


# In[138]:


df['cylinder'] = df['cylinder'].astype(int)
df.dtypes


# In[152]:


# Create subplots
numeric_columns = df.select_dtypes(include=['number']).columns
num_vars = len(numeric_columns)
fig, axes = plt.subplots(nrows=num_vars, ncols=1, figsize=(10, 5 * num_vars))

# Plot boxplots for each variable
for i, col in enumerate(numeric_columns):
    df.boxplot(column=col, ax=axes[i])
    axes[i].set_title(f'{col} Boxplot')

plt.tight_layout()
plt.show()


# In[143]:


df['engine_capacity'].value_counts()


# In[145]:


df[(df['engine_capacity'] > 100 )& (df['engine_capacity'] < 1000)]


# In[91]:


df.info()


# In[85]:


df['cylinder'] = df['cylinder'].replace('N/A, Electric', 0)
df['cylinder'].value_counts()


# In[89]:


df.info()

