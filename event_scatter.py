#Python code for generating Figure 2: Number of flood events in the lower 48 States \n (5 year average between 2012-2016)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob, os

#from mpl_toolkits.basemap import Basemap as Basemap
#from matplotlib.colors import rgb2hex
#from matplotlib.patches import Polygon
#from matplotlib import cm



tfile = open('flist.txt','w') 

for file in glob.glob("*.csv"):
	print(file)
	tfile.write(str(file)+'\n') 

tfile.close() 

events = input("Please enter your the event of interest: ")
vlist = [events]
#vlist = ['Hurricane (Typhoon)']
slist = ['ALABAMA',
'ARIZONA',
'ARKANSAS',
'CALIFORNIA',
'COLORADO',
'CONNECTICUT',
'DELAWARE',
'FLORIDA',
'GEORGIA',
'IDAHO',
'ILLINOIS',
'INDIANA',
'IOWA',
'KANSAS',
'KENTUCKY',
'LOUISIANA',
'MAINE',
'MARYLAND',
'MASSACHUSETTS',
'MICHIGAN',
'MINNESOTA',
'MISSISSIPPI',
'MISSOURI',
'MONTANA',
'NEBRASKA',
'NEVADA',
'NEW HAMPSHIRE',
'NEW JERSEY',
'NEW MEXICO',
'NEW YORK',
'NORTH CAROLINA',
'NORTH DAKOTA',
'OHIO',
'OKLAHOMA',
'OREGON',
'PENNSYLVANIA',
'RHODE ISLAND',
'SOUTH CAROLINA',
'SOUTH DAKOTA',
'TENNESSEE',
'TEXAS',
'UTAH',
'VERMONT',
'VIRGINIA',
'WASHINGTON',
'WEST VIRGINIA',
'WISCONSIN',
'WYOMING']

f = ['STATE','EVENT_TYPE']
i=0
for file in glob.glob("*.csv"):
#for i in range(16,-1,-1):
#	if i<10:
#		yr='0'+str(i)
#	else:
#		yr=str(i)
#	rstring = 'StormEvents'+yr+'.csv'
	df=pd.read_csv(file, usecols=f)
	df=df[df['EVENT_TYPE'].isin(vlist)]
	df=df[df['STATE'].isin(slist)]
	df=df.groupby(['STATE']).count()
	df.columns=[file[30:34]]
	for j in range(0,47):
		if slist[j] not in df.index:
			dftemp = pd.DataFrame([0], columns=[file[30:34]], index=[slist[j]])
			df=pd.concat([df, dftemp])
	df=df.sort_index()
	if i==0:
		full=df
	else:
		full=full.join(df,  how='outer')
		#full=df.join(full,  how='outer')
	i+=1

#x=list(range(50,99+1))+list(range(0,17+1))
#x = list(map(str, x))
# show=15
# x=list(range(0,68))
# y=full.values[show].astype(int)
# plt.scatter(x,y)
# plt.xlabel('Number of years after 1950')
# plt.ylabel('Number of Tornadoes in '+full.index[show])
# plt.show()

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression()
x=list(range(0,68))
y=full.sum().astype(int)
x=np.asarray(x)
y=np.asarray(y)
##x=np.transpose(x)
##y=np.transpose(y)
# x=x.reshape(-1,1)
# y=y.reshape(-1,1)
# regr.fit(x, y)
# y_pred = regr.predict(x)
fig = plt.figure(figsize=(10,8), dpi=100)
plt.scatter(x, y,  color='black')
# plt.plot(x, y_pred, color='blue', linewidth=3)
plt.xlabel('Years since 1950')
plt.ylabel('Total Number of '+events+' events per year \n in the lower 48 States')
#labels=['1950','1960','1970','1980','1990','2000','2010','2020']
#plt.xticks(x, labels)
#plt.rcParams["figure.figsize"] = [12,9]
#plt.figure(figsize=(20,10))
plt.show()
#print("Mean squared error: %.2f" % mean_squared_error(y, y_pred))
# print('Coefficient of determination : %.2f' % r2_score(y, y_pred))



