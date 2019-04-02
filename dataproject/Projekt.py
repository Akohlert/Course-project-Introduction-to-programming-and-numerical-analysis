import pandas as pd
#pip install linearmodels #Run in terminal
from linearmodels import PanelOLS 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas_datareader
from pandas_datareader import wb
import seaborn as sns

###### a. Downloading inflation and unemployment data from World Bank ######
cntr_eu = ['DK','SE','FR','NL','DE','GB','BE', 'LU', 'AT', 'FI'] # Subset of countries affected by ECB's QE
cntr_other = ['CA','CH','AU','NZ','SG','NO','US', 'JP', 'KR'] # Subset of countries not affected by ECB's QE

infl_eu = wb.download(indicator='FP.CPI.TOTL.ZG', country=cntr_eu, start=1991, end=2017) 
infl_other = wb.download(indicator='FP.CPI.TOTL.ZG', country=cntr_other, start=1991, end=2017)

unem_eu = wb.download(indicator='SL.UEM.TOTL.ZS', country=cntr_eu, start=1991, end=2017)
unem_other = wb.download(indicator='SL.UEM.TOTL.ZS', country=cntr_other, start=1991, end=2017)

###### b. Data structuring ######

merge_eu = pd.concat([infl_eu, unem_eu], axis=1)
merge_eu = merge_eu.reset_index() # Resetting index so "Year" can be treated as a variable instead of index
merge_eu.columns = ['country', 'year', 'inflation','unemployment'] #Naming the coloumns
merge_eu.year = merge_eu.year.astype(int) #Turning the years into integers

merge_other = pd.concat([infl_other, unem_other], axis=1)
merge_other = merge_other.reset_index() # Resetting index so "Year" can be treated as a variable instead of index
merge_other.columns = ['country', 'year', 'inflation','unemployment'] #Naming the coloumns
merge_other.year = merge_other.year.astype(int) #Turning the years into integers

#Making subset for when Quantative Easing was in effect
after_QE = merge_eu[merge_eu['year']>=2015]
after_QE_other = merge_other[merge_other['year']>=2015]

#Making subset for when Quantative Easing was in effect
after_QE = merge_eu[merge_eu['year']>=2015]
after_QE

###### c. Data description ######

#Data description for selected EU-countries
mean_infl_eu = merge_eu.groupby("country")['inflation'].mean()
min_infl_eu = merge_eu.groupby("country")['inflation'].min()
max_infl_eu = merge_eu.groupby("country")['inflation'].max()
mean_unem_eu = merge_eu.groupby("country")['unemployment'].mean()
min_unem_eu = merge_eu.groupby("country")['unemployment'].min()
max_unem_eu = merge_eu.groupby("country")['unemployment'].max()

tabel1 = pd.concat([mean_infl_eu, min_infl_eu, max_infl_eu, mean_unem_eu, min_unem_eu, max_unem_eu], axis=1)
tabel1.columns = ['Average inflation', 'Mininum inflation', 'Maximum inflation', 'Average unemployment', 'Minimum unemployment', 'Maximum unemployment']
tabel1

#Data description for selected other countries
mean_infl_other = merge_other.groupby("country")['inflation'].mean()
min_infl_other = merge_other.groupby("country")['inflation'].min()
max_infl_other = merge_other.groupby("country")['inflation'].max()
mean_unem_other = merge_other.groupby("country")['unemployment'].mean()
min_unem_other = merge_other.groupby("country")['unemployment'].min()
max_unem_other = merge_other.groupby("country")['unemployment'].max()

tabel2 = pd.concat([mean_infl_other, min_infl_other, max_infl_other, mean_unem_other, min_unem_other, max_unem_other], axis=1)
tabel2.columns = ['Average inflation', 'Mininum inflation', 'Maximum inflation', 'Average unemployment', 'Minimum unemployment', 'Maximum unemployment']
tabel2

####### d. Plots ########

#Development in inflation over time - eu countries
merge_eu.set_index('year').groupby('country')['inflation'].plot(legend=True, figsize=(12,7))
plt.title("Development in inflation 1991-2017")
plt.xlabel("Year")
plt.ylabel("Inflation")
plt.show()

#Development in unemployment over time - eu countries
merge_eu.set_index('year').groupby('country')['unemployment'].plot(legend=True, figsize=(12,7))
plt.title("Development in unemployment 1991-2017")
plt.xlabel("Year")
plt.ylabel("Unemployment")
plt.show()

#Development in inflation over time - other countries
merge_other.set_index('year').groupby('country')['inflation'].plot(legend=True, figsize=(12,7))
plt.title("Development in inflation 1991-2017 - Other")
plt.xlabel("Year")
plt.ylabel("Inflation")
plt.show()

#Development in unemployment over time - other countries
merge_other.set_index('year').groupby('country')['unemployment'].plot(legend=True, figsize=(12,7))
plt.title("Development in unemployment 1991-2017 - Other")
plt.xlabel("Year")
plt.ylabel("Unemployment")
plt.show()

#Long-run correlation between unemployment and inflation (Long run Phillips curve)
sns.set_style("whitegrid") # Setting seaborn graphstyle to "whitegrid"
LRPC_eu = sns.FacetGrid(merge_eu, col='country', hue='country', col_wrap=4, palette="deep")
LRPC_eu = LRPC_eu.map(plt.plot, 'unemployment', 'inflation').set_titles("{col_name}") 

#Short-run correlation between unemployment and inflation (SRPC) for EU-countries
SRPC_eu = sns.FacetGrid(after_QE, col='country', hue='country', col_wrap=4, palette="deep")
SRPC_eu = SRPC_eu.map(plt.plot, 'unemployment', 'inflation').set_titles("{col_name}") 

#Short-run correlation between unemployment and inflation (SRPC) for "Other" sample. 
SRPC_other = sns.FacetGrid(after_QE_other, col='country', hue='country', col_wrap=4, palette="deep")
SRPC_other = SRPC_other.map(plt.plot, 'unemployment', 'inflation').set_titles("{col_name}") 

######## e. Panel data regression analysis #######

#Panel data regression for full sample
merge_eu = merge_eu.reset_index()
year_full = pd.Categorical(merge_eu.year)
merge_eu = merge_eu.set_index(['country','year'])
merge_eu['year']=year_full
regression1=PanelOLS(merge_eu.inflation, merge_eu.unemployment, entity_effects=True)
res1 = regression1.fit(cov_type='clustered', cluster_entity=True)
print(res1)

# Panel data regression for data after QE
after_QE = after_QE.reset_index()
year_QE = pd.Categorical(after_QE.year)
after_QE = after_QE.set_index(['country','year'])
after_QE['year']=year_QE
regression2=PanelOLS(after_QE.inflation, after_QE.unemployment, entity_effects=True)
res2 = regression2.fit(cov_type='clustered', cluster_entity=True)
print(res2)

after_QE_other = after_QE_other.reset_index()
year_QE_other = pd.Categorical(after_QE_other.year)
after_QE_other = after_QE_other.set_index(['country','year'])
after_QE_other['year']=year_QE_other
regression3=PanelOLS(after_QE_other.inflation, after_QE_other.unemployment, entity_effects=True)
res3 = regression3.fit(cov_type='clustered', cluster_entity=True)
print(res3)