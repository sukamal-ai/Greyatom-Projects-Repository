# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here

#Loading the data
data=pd.read_csv(path)

#Plotting histogram of Rating
data['Rating'].plot(kind='hist')

plt.show()


#Subsetting the dataframe based on `Rating` column
data=data[data['Rating']<=5]

#Plotting histogram of Rating
data['Rating'].plot(kind='hist')   

#Code ends here
#Code ends here


# --------------
# code starts here

total_null = data.isnull().sum()

percent_null = (total_null/data.isnull().count())

missing_data = pd.concat([total_null,percent_null],axis=1,keys=['Total','Percent'])

print(missing_data)

data = data.dropna()

total_null_1 = data.isnull().sum()

percent_null_1 = (total_null/data.isnull().count())

missing_data_1 = pd.concat([total_null_1,percent_null_1],axis=1,keys=['Total','Percent'])

print(missing_data_1)


# code ends here


# --------------

#Code starts here

plt.figure(figsize=(5,5))
sns.catplot(x="Category",y="Rating",data=data,kind="box", height=10)
plt.xticks(rotation=90)
plt.title("Rating vs Category [Boxplot]")

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here

data['Installs'] = data['Installs'].str.replace(",","")
data['Installs'] = data['Installs'].str[:-1]
data['Installs'] = data['Installs'].astype('int')

print(data['Installs'])

from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder()

le.fit(data['Installs'])
data['Installs'] = le.transform(data['Installs'])

sns.regplot(x="Installs",y="Rating",data=data)
plt.title("Rating vs Installs[RegPlot]")



#Code ends here



# --------------
#Code starts here
import numpy as np

data['Price'] = data['Price'].str.replace("$","")
data['Price'] = data['Price'].astype(float)
print(data['Price'].value_counts())

sns.regplot(x="Price",y="Rating",data=data)
plt.title("Rating vs Price[Regplot]")
#Code ends here


# --------------

#Code starts here

def mask(data):
    l = data.split(";")
    return l[0]

data['Genres'] = data['Genres'].apply(mask)

gr_mean = data[['Genres','Rating']].groupby('Genres',as_index=False).mean()

gr_mean.describe()


gr_mean = gr_mean.sort_values(by="Rating")

print(gr_mean.head(1))
print(gr_mean.tail(1))
#Code ends here


# --------------

#Code starts here



data['Last Updated'] = pd.to_datetime(data['Last Updated'])

max_date = data['Last Updated'].max()

data['Last Updated Days'] = max_date - data['Last Updated']

data['Last Updated Days'] = data['Last Updated Days'].dt.days

sns.regplot(x="Last Updated Days", y="Rating",data=data)
#Code ends here


