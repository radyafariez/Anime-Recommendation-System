#!/usr/bin/env python
# coding: utf-8

# # **RECOMMENDER SYSTEM - ANIME MOVIES & SERIES**

# ### **Deskripsi**: Membuat sistem rekomendasi konten movie & series anime menggunakan Content Based Filtering

# In[2]:


get_ipython().system('pip install opendatasets')
get_ipython().system('pip install pandas')


# **Import Library yang dibutuhkan**

# In[6]:


import pandas as pd
import opendatasets as od
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# **Install Dataset melalui Kaggle Repositories**

# In[7]:


#! pip install kaggle
#! mkdir ~/.kaggle
#! cp kaggle.json ~/.kaggle
#! chmod 600 ~/.kaggle/kaggle.json
#! kaggle datasets download -d CooperUnion/anime-recommendations-database
od.download(
    "https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database"
    )


# Unzip dataset

# In[10]:


#! unzip anime-recommendations-database.zip


# Mendefinisikan dataset CSV dan listing genre

# In[12]:


anime = pd.read_csv('anime.csv')

values = ['Drama', 'Romance', 'Adventure', 'Action', 'SciFi', 'Fantasy', 'Comedy', 'Slice of Life', 'Sports', 'Horror', 'Supernatural', 'Mystery', 'Historical']
animes = anime[anime.genre.isin(values) == True]
animes.sample(3)


# In[13]:


anime.head(3)


# In[15]:


animes.head(3)


# In[14]:


ratings = pd.read_csv('rating.csv')
ratings.sample(3)


# **Deskripsi Dataset**

# **Memperoleh informasi mengenai dataset 'anime.csv' dan 'ratings.csv'**

# Mengetahui jumlah baris dan kolom pada data

# In[16]:


animes.shape


# In[17]:


ratings.shape


# In[18]:


animes.info


# In[19]:


ratings.info


# Mengetahui rentang nilai rating pada data

# In[20]:


ratings.describe()


# Menghitung jumlah/overall masing - masing konten anime dan genre

# In[21]:


print('Jumlah Anime: ', len(animes.anime_id.unique()))
print('Jumlah Genre: ', len(animes.genre.unique()))


# Memperoleh jenis genre dan tipe penayangan konten anime

# In[22]:


print('Genre: ', animes.genre.unique())
print('Types: ', animes.type.unique())


# **Exploratory Data Analysis**

# Visualisasi data, untuk mengetahui porsi jumlah masing - masing genre terhadap keseluruhan data

# In[23]:


categorical_features = ['genre']
feature = categorical_features[0]
count = animes[feature].value_counts()
percent = 100*animes[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah Anime':count, 'Persentase':percent.round(1)})
print(df)


# In[24]:


count.plot(kind='bar', title=feature);


# Menghitung jumlah user ID dan data rating berdasarkan raw dataset saat ini

# In[25]:


print('Jumlah User: ', len(ratings.user_id.unique()))
print('Jumlah rating: ', len(ratings))


# In[26]:


ratings.hist(bins=50, figsize=(20,15))
plt.show()


# **Data Preparation**

# **Menggabungkan data anime_id pada dataset 'anime' dan 'ratings' untuk memperoleh info jumlah aktual pada fitur anime_id**

# In[27]:


animes_allid = np.concatenate((
    animes.anime_id.unique(),
    ratings.anime_id.unique(),
))
 
animes_allid = np.sort(np.unique(animes_allid))
 
print('Jumlah seluruh data id berdasarkan anime_id: ', len(animes_allid))


# **Menggabungkan data rating pada dataset 'anime' dan 'ratings' untuk memperoleh info jumlah aktual pada fitur rating**

# In[28]:


animes_allrate = np.concatenate((
    animes.rating.unique(),
    ratings.rating.unique(),
))
 
# Mengurutkan data
animes_allrate = np.sort(np.unique(animes_allrate))
 
print('Jumlah seluruh data rating berdasarkan anime_id: ', len(animes_allrate))


# Memeriksa apakah terdapat data yang kosong atau 'NaN' pada data

# In[29]:


animes.isnull().sum()


# **Menghapus data yang kosong atau 'NaN'**

# In[30]:


animes_fix = animes.dropna()
animes_fix


# **Sorting data/mengurutkan data dari angka terkecil fitur anime_id**

# In[31]:


animes_fixed = animes_fix.sort_values('anime_id', ascending=True)
animes_fixed


# In[32]:


print('Jumlah anime_id: ', len(animes_fixed.anime_id.unique())) 
print('Jumlah genre: ', len(animes_fixed.genre.unique()))


# **Drop value yang memiliki data duplikat pada dataset yang telah diolah sebelumnya**

# In[33]:


preparation = animes_fixed.drop_duplicates('anime_id')
preparation


# **Melaukan konversi data pada masing - masing fitur menjadi format/bentuk list**

# In[34]:


# Mengonversi data series 'animeID'
animeID = preparation['anime_id'].tolist()
 
# Mengonversi data series ‘title’ 
title = preparation['name'].tolist()
 
# Mengonversi data series ‘genres’
genres = preparation['genre'].tolist()
 
print(len(animeID))
print(len(title))
print(len(genres))


# **Membuat data dictionary**

# In[35]:


anime_new = pd.DataFrame({
    'id': animeID,
    'titles': title,
    'genre': genres
})
anime_new     


# In[36]:


# Melihat sampel data
data = anime_new
data.sample(5)


# **Modelling**

# **Inisialisasi TfidfVectorizer, melakukan perhitungan idf dan mapping array**

# In[37]:


tf = TfidfVectorizer()
 
tf.fit(data['genre']) 
 
# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names() 


# **Melakukan fitment dan transformasi ke bentuk matrix**

# In[38]:


tfidf_matrix = tf.fit_transform(data['genre']) 

# Melihat jumlah baris & data matrix tfidf
tfidf_matrix.shape 


# Mengubah vektor Tfidf dalam bentuk matriks dengan fungsi todense

# In[39]:


tfidf_matrix.todense()


# Membuat dataframe dengan konten baris berisi fitur titles dan kolom berisi fitur genre

# In[40]:


pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names(),
    index=data.titles
).sample(12, axis=1).sample(10, axis=0)


# **Menghitung cosine similarity**

# In[41]:


cosine_sim = cosine_similarity(tfidf_matrix) 
cosine_sim


# Membuat dataframe dan meihat cosine similarity

# In[42]:


cosine_sim_df = pd.DataFrame(cosine_sim, index=data['titles'], columns=data['titles'])
print('Shape:', cosine_sim_df.shape)
 
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)


# **Mengambil data dengan menggunakan argpartition**

# In[43]:


def animes_recommendations(titles, similarity_data=cosine_sim_df, items=data[['titles', 'genre']], k=10): 
    # Dataframe -> numpy
    # Range start, stop, step
    index = similarity_data.loc[:,titles].to_numpy().argpartition(
        range(-1, -k, -1))
    
    # Mengambil data dengan similarity terbesar
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # drop titles agar nama konten yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(titles, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)


# **Result**

# **Menguji model sistem rekomendasi**

# In[44]:


titles = 'Wolf Daddy'
data[data.titles.eq(titles)]


# Hasil rekomendasi yang didapatkan

# In[45]:


titles = 'Wolf Daddy'
animes_recommendations = animes_recommendations(titles)
animes_recommendations

