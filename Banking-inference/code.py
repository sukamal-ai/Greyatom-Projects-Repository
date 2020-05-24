# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

data = pd.read_csv(path)

data_sample = data.sample(n=sample_size,random_state=0)

#Code starts here

sample_mean = data_sample['installment'].mean()

sample_std = data_sample['installment'].std()

margin_of_error = z_critical * (sample_std/math.sqrt(sample_size))

confidence_interval = (sample_mean-margin_of_error, sample_mean+margin_of_error)

true_mean = data['installment'].mean()

print(true_mean)
print(confidence_interval)






# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
fig, axes = plt.subplots(3,1)

for i in range(len(sample_size)):
    m = []
    for j in range(1000):
        sample_data = data.sample(n=sample_size[i])
        m.append(sample_data['installment'].mean())

    mean_series = pd.Series(m)
    axes[i] = mean_series.hist()


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest
import numpy as np

#Code starts here
data['int.rate'] = data['int.rate'].str[:-1]

data['int.rate'] = data['int.rate'].astype(np.float64) / 100

x1 = data[data['purpose']=='small_business']['int.rate']

value = data['int.rate'].mean()

z_statistic, p_value = ztest(x1,value=value,alternative='larger')

print(z_statistic)
print(p_value)


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here

x1 = data[data['paid.back.loan']=='No']['installment']
x2 = data[data['paid.back.loan']=='Yes']['installment']

z_statistic, p_value = ztest(x1,x2)


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here

yes = data[data['paid.back.loan']=='Yes']['purpose'].value_counts()

no = data[data['paid.back.loan']=='No']['purpose'].value_counts()

observed = pd.concat([yes.transpose(), no.transpose()], axis=1,keys=['Yes','No'])

chi2, p, dof, ex = chi2_contingency(observed)


