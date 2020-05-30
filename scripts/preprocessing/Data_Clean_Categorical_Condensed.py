#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


homes = pd.read_csv('train.csv')


# In[4]:


ms_zone_cat = ['FV','RL']
df_ms_zone_cat = pd.DataFrame(ms_zone_cat)
df_ms_zone_cat['MS_Zoning_group'] = 'FV/ RL'
df_ms_zone_cat = df_ms_zone_cat.rename(columns={0:'MSZoning'})
homes = pd.merge(homes,df_ms_zone_cat,how='left',on= 'MSZoning')
homes.MS_Zoning_group = homes.MS_Zoning_group.fillna('Other')


# In[5]:


homes = homes.drop(columns = ['Street'])


# In[6]:


lotshape_cat = ['Reg']
df_lotshape_cat = pd.DataFrame(lotshape_cat)
df_lotshape_cat['LotShape_group'] = 'Reg'
df_lotshape_cat = df_lotshape_cat.rename(columns={0:'LotShape'})
homes = pd.merge(homes,df_lotshape_cat, how = 'left', on = 'LotShape')
homes.LotShape_group = homes.LotShape_group.fillna('Irregular')





# In[8]:


homes.Alley = homes.Alley.fillna('No_alley_acess')


# In[9]:


landcontour_ = {'LandContour': ['Lvl','Bnk','HLS','Low'],'LandContour_group' : ['Lvl','Bnk','Other','Other']}
df_landcontour_cat = pd.DataFrame(data= landcontour_)






# In[ ]:




