# --------------
#Importing header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Code starts here
data = pd.read_csv(path)

loan_status = data['Loan_Status'].value_counts()

print(loan_status)

loan_status.plot(kind="bar")


# --------------
#Code starts here

property_and_loan = data.groupby(['Property_Area','Loan_Status'])

property_and_loan = property_and_loan.size().unstack()

plot = property_and_loan.plot(kind='bar',stacked=False,rot=45)

plot.set_xlabel('Property Area')
plot.set_ylabel('Loan Status')



# --------------
#Code starts here

education_and_loan = data.groupby(['Education','Loan_Status']).size().unstack()

plot = education_and_loan.plot(kind='bar',stacked=True,rot=45)

plot.set_xlabel('Education Status')
plot.set_ylabel('Loan Status')




# --------------
#Code starts here

graduate = data[data['Education']=='Graduate']

not_graduate = data[data['Education']=='Not Graduate']

graduate.plot(kind='density',label='Graduate')

not_graduate.plot(kind='density',label='Not Graduate')





#Code ends here

#For automatic legend display
plt.legend()


# --------------
#Code starts here
fig, (ax_1, ax_2,ax_3) = plt.subplots(3,1, figsize=(20,40))

data['ApplicantIncome'].fillna(0)
data['CoapplicantIncome'].fillna(0)

data['TotalIncome'] = data['ApplicantIncome'] + data['CoapplicantIncome']

data.plot.scatter(x='ApplicantIncome',y='LoanAmount',ax=ax_1)

data.plot.scatter(x='CoapplicantIncome',y='LoanAmount',ax=ax_2)

data.plot.scatter(x='TotalIncome',y='LoanAmount',ax=ax_3)




