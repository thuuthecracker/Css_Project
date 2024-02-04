#!/usr/bin/env python
# coding: utf-8

# In[2]:


# In[3]:


import pandas as pd
file = pd.read_csv("movie_dataset.csv",index_col=0)
file


# # Filling up the missing data

# In[4]:


x = file["Revenue (Millions)"].mean()

file["Revenue (Millions)"].fillna(x, inplace = True) 


y = file["Metascore"].mean()

file["Metascore"].fillna(y, inplace = True)

file


# # What is the average revenue of all movies in the dataset?

# In[5]:


avg=sum(file["Revenue (Millions)"])/len(file["Revenue (Millions)"])
print(avg)


# # What is the average revenue of movies from 2015 to 2017 in the dataset?

# In[6]:


t= file[file["Year"]>=2015]


avg=sum(t["Revenue (Millions)"])/len(t["Revenue (Millions)"])
print(avg)


# # How many movies were released in the year 2016?

# In[7]:


z=file[file["Year"]==2016]
len(z)


# # How many movies were directed by Christopher Nolan?

# In[8]:


z=file[file["Director"]=='Christopher Nolan']
len(z)


# # How many movies in the dataset have a rating of at least 8.0?

# In[9]:


z=file[file["Rating"]>=8]
len(z)


# # What is the median rating of movies directed by Christopher Nolan?

# In[10]:


z=file[file["Director"]=='Christopher Nolan']
zz=z.mean()
zz #The answer is 8.68


# # Find the year with the highest average rating?

# In[11]:


z=file[file["Year"]==2006]
zz=z.mean()
print(zz)


# In[12]:


z=file[file["Year"]==2007]
zz=z.mean()
print(zz)


# In[13]:


z=file[file["Year"]==2008]
zz=z.mean()
print(zz)


# In[14]:


z=file[file["Year"]==2009]
zz=z.mean()
print(zz)


# In[15]:


z=file[file["Year"]==2010]
zz=z.mean()
print(zz)


# In[16]:


z=file[file["Year"]==2011]
zz=z.mean()
print(zz)


# In[17]:


z=file[file["Year"]==2012]
zz=z.mean()
print(zz)


# In[18]:


z=file[file["Year"]==2013]
zz=z.mean()
print(zz)


# In[19]:


z=file[file["Year"]==2014]
zz=z.mean()
print(zz)


# In[20]:


z=file[file["Year"]==2015]
zz=z.mean()
print(zz)


# In[21]:


z=file[file["Year"]==2016]
zz=z.mean()
print(zz)


# # Answer is 2007

# # What is the percentage increase in number of movies made between 2006 and 2016?

# In[22]:


z=file[file["Year"]==2006]
zz=file[file["Year"]==2016]

t=((len(zz)-len(z))/(len(z)))*100
t


# # Find the most common actor in all the movies?
# 
# Note, the "Actors" column has multiple actors names. You must find a way to search for the most common actor in all the movies.

# In[23]:


actors = file['Actors'].str.split(', ', expand=True).stack()
actor_counts = actors.value_counts()
most_common_actor = actor_counts.idxmax()
print("The most common actor in all the movies is:", most_common_actor)


# # How many unique genres are there in the dataset?
# 
# Note, the "Genre" column has multiple genres per movie. You must find a way to identify them individually.

# In[24]:


unique_genres = file['Genre'].str.split(',', expand=True).stack().unique()
num_unique_genres = len(unique_genres)
print("Number of unique genres:", num_unique_genres)


# # Do a correlation of the numerical features, what insights can you deduce? Mention at least 5 insights.
# 
# And what advice can you give directors to produce better movies?

# In[26]:


correlation_matrix = file.corr()
print(correlation_matrix)


# # Insights:
# # 1. Positive correlation values close to 1 indicate strong positive linear relationships between features and it is shown in runtime in minutes
# # 2. Negative correlation values close to -1 indicate strong negative linear relationships as seen in votes.
# # 3. Correlation close to 0 suggests weak or no linear relationship between features as seen metascore
# # 4. Highly correlated features can help in feature selection for machine learning models as seen in runtime
# # 5. Most of the negative correlation is observed in years
# # Advice: Go for the best actors
# 

# In[ ]:





# In[ ]:





# In[ ]:




