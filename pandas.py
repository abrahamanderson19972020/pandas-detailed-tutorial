# Pandas is a library that provides data structures and and data analysis code within Python
#Pandas allow us to load data from different sources into python and then use Python code to analyze those data
# and produce results which can be in the form of tables,text,and visiluasition with the help of visualization libraries such as Bokeh,
#its key data structure is called the DataFrame.
#DataFrames allow you to store and manipulate tabular data in rows of observations and columns of variables.

dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

import pandas as pd
brics = pd.DataFrame(dict) ## Converting the dictionary into DataFrame 
#A data frame is comprised of columns and rows.
print(brics) #This will return as below:
#        country    capital    area  population
#0        Brazil   Brasilia   8.516      200.40
#1        Russia     Moscow  17.100      143.50
#2         India  New Dehli   3.286     1252.00
#3         China    Beijing   9.597     1357.00
#4  South Africa   Pretoria   1.221       52.98

#As we can see above with the new brics DataFrame,Pandas has assigned a key for each country as the numerical values 0 through 4. 
#If we would like to have different index values, say, the two letter country code, 
#we can do that easily as well as below:
brics.index=["BR","RUS","IND","CHN","SAFR"]
print(brics) #It will be seen as:
#           country    capital    area  population
#BR          Brazil   Brasilia   8.516      200.40
#RUS         Russia     Moscow  17.100      143.50
#IND          India  New Dehli   3.286     1252.00
#CHN          China    Beijing   9.597     1357.00
#SAFR  South Africa   Pretoria   1.221       52.98


#Pandas DataFrame is two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns).
#Pandas DataFrame consists of three principal components, the data, rows, and columns.
#a Pandas DataFrame will be created by loading the datasets from existing storage,  
#storage can be SQL Database, CSV file, and Excel file. 
#We can also create a DataFrame>
#1: Create a Pandas DataFrame from Lists
#1.1: Creating one dimensional lists:
import pandas
lst=["Abraham","Naim","Pinar","Kader"]
df=pandas.DataFrame(lst)
df.columns=["Name"]
df.index=["Number 1", "Number2","Number3","Number4"]
#we can do the same via:
#df = pd.DataFrame(lst, index =['number1', 'number2', 'number3', 'number4'], columns =['Names'])
print(df) #This will be seen as below:
#            Name
#Number 1  Abraham
#Number2      Naim
#Number3     Pinar
#Number4     Kader

#1.2:Creating multidimensional lists:
lst=[["Ibrahim","1983"],["Kader","1988"],["Naim","2012"],["Pinar","2016"]]
df=pandas.DataFrame(lst)
df.columns=["Name","Date of Birth"]
print(df) #This will return with two columns because we every item in the list has two items


import pandas 
df1=pandas.DataFrame([[2,4,5],[10,20,30]])
print(df1)#This will be seen as:
#   0   1   2
#0   2   4   5
#1  10  20  30

df1=pandas.DataFrame([[2,4,5],[10,20,30]],columns=["Price","Price","Age"],index=["First","Second"])
print(df1)#This will be seen as:
#Price  Price  Age
#First       2      4    5
#Second     10     20   30

df2=pandas.DataFrame([["Name","Surname"],["Ibrahim","Guvenc"],["Kader","Guvenc"]])
print(df2)

dir(pandas.DataFrame()) #This returns all the methods for analyzing DataFrames in pandas
country={"Norway":[500000,75000],"Turkey":[80000000,10000],"China":[2000000000,12000]}
df=pandas.DataFrame(country)
df.index=["Population","GDP"]
print(df)
print(df.Norway.mean()) #gives the mean of the data below Norway column
print(df.China.max()) #returns the max valua under China column

import pandas
data={"Names:":["Naim","Kader","Sevim","Pinar"], "Grades":[100,90,90,100],"Birthdate":[2012,1987,1963,2016]}
df=pandas.DataFrame(data)
print(df)
df.columns=["Name","Grade","Date of Birth"]
print(df)
print(df.loc[0])
print(df[["Grade","Name"]])

# 2:  How to Access Last and First Rows of DataFrame in Pandas:
#.head() returns the rows from the beginning according to value inside ()
print(df.head(1)) # This returns the forst row as follows:
 #Name  Grade  Date of Birth
#0  Naim    100           2012
print(df.head(2)) #This returns the first two rows
#   Name  Grade  Date of Birth
#0   Naim    100           2012
#1  Kader     90           1987

#.tail() returns the rows by beginning from the last row
print(df.tail(1)) # this returns the last row
#    Name  Grade  Date of Birth
#3  Pinar    100           2016
print(df.tail(2)) # This returns the last two rows
#Name  Grade  Date of Birth
#2  Sevim     90           1963
#3  Pinar    100           2016


# 5:  How to Change the Column in Pandas DataFrame?

import os
os.listdir()

#reading csv,json,excel and txt files


import pandas
df=pandas.read_csv("immigration.csv")
print(df)

import pandas
df1=pandas.read_json("immigration.json")
print(df1)

df2=pandas.read_excel("immigration.xlsx")
print(df2)

with open("immigration.txt") as myfile:
    file=myfile.read()
print(file)

#Indexing and Slicing in Pandas
df=pandas.read_json("immigration.json")
print(df)
print(".......................")
#.loc["rowx":"rowy", "columnx":"columny"] returns the output between mentioned rows and columns
# .loc[] provides us label based indexing and slicing
x=df.loc["1":"4","Immigrant Population in Norway":"Popularion in Norway"]
print(x) #this will return as follows:
#Immigrant Population in Norway Popularion in Norway
#1                        246,938            4,405,158
#2                        305,035            4,513,747
#3                        361,143            4,623,293
#4                        488,753            4,828,716
#we can also find out one intersection between a single row and column as follows:
y=df.loc["2":"5","Population in Sweden"]
print(y) #this wil return the Population in  Sweden column between row 2 and 5
z=df.loc[:],"Popularion in Norway"
print(z)#This will return all the row for Population in Norway column


#.iloc[] is more common indexing and it is not a label based indexing and slicing
# we give the index of the data instead of the names of the data as it is case in .loc[]
x=df.iloc[0:2, 1:3] #this returns tha data between row0 and row 1 and column 1 and column 2
print(x) #this will return as follows:
#Immigrant Population in Norway Popularion in Norway
#0                        195,700            4,286,397
#1                        246,938            4,405,158
#.iloc[ ]is better way for indexing and slicing because we do not need to write the long names of the data
y=df.iloc[:,2:5] #this returns all the rows between mentioned columns
print(y)
print("..........")
z=df.iloc[1:3, :]#this will return all the columns between row 1 and row 2
print(z)
print("..........")
d=df.iloc[1,3] #this will return the between row 1 and column 3
print(d)



#Deleting Rows and Columns in Pandas:
# we use .drop("columnname",1) in order to delete a column:
df=df.drop("Population in Sweden",1)
print(df)
# we use .drop("row name", 0) in order to delete a row
print("...........")
df=df.drop("1997",0)
print(df)



#Adding and updating new columns and rows:
#we can add new columns as variable["name"]=["it must have items as many as the length of the rows"]

df["Continent"]=df.shape[0]*["Europa"]
print(df)# this will return as follows:
#FIELD1 Immigrant Population in Norway  ... Population in Sweden Continent
#0    1992                        195,700  ...            8,668,065    Europa
#1    1997                        246,938  ...            8,846,059    Europa
#2    2001                        305,035  ...            8,895,963    Europa
#3    2005                        361,143  ...            9,029,567    Europa
#4    2009                        488,753  ...            9,298,510    Europa
#5    2013                        663,870  ...            9,600,375    Europa
#6    2017                        799,797  ...           10,057,695    Europa
df["Size"]=["121092","232432","0345930","940954","94595","945854","45909"]
#here we assign which item will be included in the new column
print(df) #this will return as follows:
#FIELD1 Immigrant Population in Norway  ... Continent     Size
#0    1992                        195,700  ...    Europa   121092
#1    1997                        246,938  ...    Europa   232432
#2    2001                        305,035  ...    Europa  0345930
#3    2005                        361,143  ...    Europa   940954
#4    2009                        488,753  ...    Europa    94595
#5    2013                        663,870  ...    Europa   945854
#6    2017                        799,797  ...    Europa    45909
df["Continent"]=["Europa","North America","South America","Africa","Asia","Antartica", "Ocenia"]
print(df)
#in order to add new row, we have firstly transpose the data and interchange rows and coluns
# in order to transpose row and columns we use .T
df1=df.T #the rows will be columns of the new data and vice versa
print(df1)#this will return as follows :
#0              1  ...          5           6
#FIELD1                               1992           1997  ...       2013        2017
#Immigrant Population in Norway    195,700        246,938  ...    663,870     799,797
#Popularion in Norway            4,286,397      4,405,158  ...  5,109,056   5,295,619
#Immigrant Population in Sweden    814,200        943,804  ...  1,473,256   1,784,497
#Population in Sweden            8,668,065      8,846,059  ...  9,600,375  10,057,695
#Continent                          Europa  North America  ...  Antartica      Ocenia
#Size                               121092         232432  ...     945854       45909
#After that we use the same methods 
print(len(df1)) # we learn that df1 have 7 rows , so we have to write 7 items for the new column
df1["7"]=["2018","800,000", "300,000","9232093","10,202,009","Europa","34343"]
print(df1)
df=df1.T #here after adding new column, we transpose again and new column become new row
print(df)


""" Extracting essential data from a dataset and displaying it is a necessary part of data science; 
therefore individuals can make correct decisions based on the data. Here is an example of extracting data from URL:"""
import pandas as pd
links={'GDP':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_gdp.csv',\
       'unemployment':'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/PY0101EN/projects/coursera_project/clean_unemployment.csv'}
df=pd.read_csv(links["GDP"])
# Use the method head() to display the first five rows of the GDP data, then take a screen-shot.
print(df.head(5))
#Use the dictionary links and the function pd.read_csv to create a Pandas dataframes that contains the unemployment data.
df_unemp=pd.read_csv(links["unemployment"])
print(df_unemp)
#Use the method head() to display the first five rows of the GDP data, then take a screen-shot.
print(df_unemp.head(5))
#Question 3: Display a dataframe where unemployment was greater than 8.5%
for item in df_unemp["unemployment"]:
    if item > 8.5:
        print(item)



""" read_csv assumes that the data we enter has column headers. However if not, we have to specify as header=None after the url"""
import pandas as pd
df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",header=None)
# pandas automatically sets the header column names as default if we do not specify with number begining with 0.


"""Rather than seeing all the data we can select the first rows with df.head(n),or the last rows via df.tail(n) methods of pandas"""
x= df.head(5) #returns the first 5 rows
y=df.tail(3) # return the last 3 row of the data
print(x)

""" if we want to export our pandas data frame into another file, we do it with to_csv, to_xlsx() method """

# path="C:\Users\ibrahim\Desktop\Pythonfiles"
# df.to_csv(path)
# we use df.dtypes to check the data types
# print(df.dtypes) #This returns the data type of each column

# dataframe.describe() method returns a statistical summary of the only columns which includes numerical values
#like the mean, standart deviation, min, max values, number if items in the column etc are listed via df.describe() method
print(df.describe()) # this methods skips rows and columns that do not contain numbers.
#If we want to see all of the columns and rows including non-numerical ones,we should add include="all" inside method
print(df.describe(include="all")) # The non-numerical columns shows as NaN
#This will add other statistical sets are included like uniques,top,frequency in addition to mean,std,max and min values
#Unique:is the number of distinct object in the colund
#Top: is the most frequently recurrent object, that is the mode of the column
#freq:is the number of times the top object appears in the column

# dataframe.info() method provides a concise summary of our dataframe
print(df.info())

""" How to deal with misisng values
    1.we can remove the data where the missing values is found
    2.replacing the missing values we can replace the overage value of the entire variable 
      If it is a categorical variable and si impossible find an average value, we can use mode, the most common value,
    3. we can leave the data as it is"""
import pandas as pd
# to remove the data that contains missing value, we use dataframe.dropna() method
my_dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }
df=pd.DataFrame(my_dict)
print(df)
df.dropna(subset="country",axis=1, inplace=True) # drops all columns that contain a nan
print(df)
#In order to replace values we can use df.replace(np.nan,mean)
mean = df["normalized-losses"].mean() 
df["normalized-losses"].replace(np.nan, mean)
df['peak-rpm'].replace(np.nan, 5,inplace=True)#replace the not a number values with 5 in the column 'peak-rpm'

avg=df['horsepower'].mean(axis=0)
df['horsepower'].replace(np.nan, avg)
#calculate the mean value for the 'horsepower' column and replace all the NaN values of that column by the mean value

df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)
#rename column name from "highway-mpg" to "highway-L/100km"



""" Data Formatting: biringing data into a common standart of expression in order to make meaningful comparisons
# Data types in pandas:
    1.Object=string
    2.Int64=int
    3.Float64=float
# To indentify data type, we can use dataframe.dtypes()"""
import pandas as pd
my_dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 520.98] }
df=pd.DataFrame(my_dict)
print(df)
print(df.dtypes) #this returns all of the types column by column as follows:
#country        object
#capital        object
#area          float64
#population    float64
#dtype: object

#In order to convert data type, we use dataframe.astype() method
print(df["area"].dtype) # it is float64 before conversion
df["area"]=df["area"].astype("int")
print(df["area"])#it is in32 after conversion

""" Data Normalization: we do it to comprate data and make regression between several columns which have different ranges.
    #There are several approaches for normalization:
        1:Simple Feature Scaling= xnew=x old/x nmax (this make new value between 0 and 1)
        2:Min-Max Sacling: x new= x old-xmin/ xmax-xmin
        3: Z-score=xnew=xold-mean/standart deviation which typically ranges between -3 and +3 """

#The First Approach:
df["area"]= df["area"]/df["area"].max()
df["population"]=df["population"]/df["population"].max()
print(df["area"],df["population"])
# df["area"]                    df["population]
#0    0.470588                    0.105748  
#1    1.000000                    0.922623
#2    0.176471                    1.000000
#3    0.529412                    0.383920
#4    0.058824

#The Second Approach:
df["area"]=(df["area"]-df["area"].min())/(df["area"].max()-df["area"].min())
df["population"]=(df["population"]-df["population"].min())/(df["population"].max()-df["population"].min())
print(df["area"],df["population"])

#The Third Approach(Z-score)
df["area"]=(df["area"]-df["area"].mean())/df["area"].std()
df["population"]=(df["population"]-df["population"].mean())/df["population"].std()
print(df["area"],df["population"])
# as seen above it is much easier to compare different values of different columns within same value standarts


"""Binning or Data Groupering: group values into bins
  for example we can group age from 0 to 5, 6-10, 10-15, 15-20, within a categories or groups to make data more readable or comparable
    # proces=[5000,10000,12000,...,30000,31000,...44000,45000]
    we can vategorize them as low price(between 5000-20000); medium price(20000-35000; high price(36000-45000))
    here we categorize all of these values into just three bins or group or categories
  
    """
import pandas as pd
import numpy as np
df=pd.read_excel("immigration.xlsx") #here we read out excel file with pandas read_excel() method
print(df["Population in Sweden"].min())
#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)[source]
#Return evenly spaced numbers over a specified interval.
bins=np.linspace(df["Population in Sweden"].min(),df["Population in Sweden"].max(),4)
group_names=["Low","Medium","High"]
#pd.cut() method is used to segment and sort the data values into bins
df["binned"]=pd.cut(df["Population in Sweden"],bins,labels=group_names,include_lowest=True)
print(df["binned"])
# This will return all the number in the column Population in Sweeden with binned values as low,medium and high as follow:
#0       Low
#1       Low
#2       Low
#3       Low
#4    Medium
#5      High
#6      High
#Name: binned, dtype: category
#Categories (3, object): [Low < Medium < High]
import matplotlib.pyplot as plt
plt.plot(df["binned"])
plt.show() #here we can show in a graph

""" dataframe.groupby() method is used for categorical variables and group data into subsets
    we can groupby() single or multiple variables by passing in multiple variable names """
import pandas as pd
df=pd.read_excel("testdata.xls")
print(df.columns)
df.rename(columns={"Average score (grade)":"Grade"},inplace=True)
print(df.columns)
df_set=df[["Gender","Major","Grade"]] #here we pick the data we want to group by
print(df_set)
df_group=df_set.groupby(["Gender","Major"])
#When you use as_index=False, you indicate to groupby() that you don't want to set the column ID as the index (duh!). 
df_group=df_set.groupby(["Gender","Major"],as_index=False).mean() 
# here we group data into subcategories and only the average grade of each suncategory is shown
print(df_group) 
#Gender     Major      Grade
#0  Female      Econ  70.333333
#1  Female      Math  79.040322
#2  Female  Politics  84.605890
#3    Male      Econ  78.666968
#4    Male      Math  83.056642
#5    Male  Politics  85.583209
# here we can see that male in politics major have the highest grades across genr-major subgroups
"""We can transform this data into a pivot table via pivot() method in order to analyze the data better
How would you use the groupby function to find the average "price" of each car based on "body-style" ?
df[['price','body-style']].groupby(['body-style'],as_index= False).mean() """
#A pivot table is a table of statistics that summarizes the data of a more extensive table 
#via pivot() mehtod, one variable can be displayed along the columns and the other is displayed along the rows.
df_pivotted=df_group.pivot(index="Gender", columns="Major")
print(df_pivotted) #this returns as follows:
#            Grade                      
#Major        Econ       Math   Politics
#Gender                                 
#Female  70.333333  79.040322  84.605890
#Male    78.666968  83.056642  85.583209
"""Heatmap plot is another way to represent the pivot table
    It is a great way to plot the target/independent variable over multiple dependent variables through visual clues
    pyplot.pcolor() method plot heat map and covert previus pivot table into a graphical form"""
import matplotlib.pyplot as plt
plt.pcolor(df_pivotted,cmap="RdBu") #cmap="RdBu" specify the red blue color scheme
plt.colorbar()
plt.show()

"""Turning Categorical Variables into Quantitative Variable in Python 
    #Most statistical models can not take in strings/objects as input, just focus on numerical data
    # so we need to convert categorical variables into numeric format by adding new features we would like to encode
    # we add dummy variables for each unique category
    # we add numeric values for each category
# we use pandas.get_dummies() method in order to convert categorical variables into numeric dummy ones
"""
import pandas as pd
df=pd.read_csv("supermarkets.csv")
df["Bensin"]=["Diesel","Gas","Fuel","Gas","Diesel","Diesel"]
print(df)
print(pd.get_dummies(df["Bensin"]))
# pd.get_dummies() method automatically generates a list of number each one correspond particular category:
#0       1     0    0
#1       0     0    1
#2       0     1    0
#3       0     0    1
#4       1     0    0
#5       1     0    0

"""PANDAS:
    *Pandas is a open source library built on NumPy
    *It allows us fast analysis,data cleaning and preparation
    *It is also called python's version of excel
    *It has also builtin data visualization features
    
    1:Series: is very similar to numpy arrays, the difference is series have their index for each value.there are different kinds of series:"""
import numpy as np
import pandas as pd
labels=["a","b","c"]
my_data=[10,20,30]
arr=np.array(my_data)
dic={"a":10,"b":20,"c":30}
# in order to create a serie we use pd.Series() function
"""pandas.Series() can take a variety of objects: """
ser=pd.Series(my_data)
print(ser) # this returns as follows:
#0    10   # in series, every value is shown with its index in the serie
#1    20
#2    30
#dtype: int64
# we can also specify what the index in the serie will be
print(pd.Series(data=my_data,index=labels)) # here we change the label of index from 0,1,2 to a.,b,c. This returns as follows:
#a    10
#b    20
#c    30
#dtype: int64
print(arr) # this numpy array returns as [10 20 30]
print(pd.Series(arr)) #If we insert numpy array inside this functions, it will return the same output as ordinary lists as follows:
#0    10
#1    20
#2    30
#dtype: int32
# we can also insert dictinaries inside pd.Series() function and the keys will become the index of the serie and value as value
print(pd.Series(dic))
#a    10
#b    20
#c    30
#dtype: int64
# we can also create series with different ways:
ser1=pd.Series([1,2,3,4],["USA","China","Russia","England"]) # here we assign countries as the labels of index of values
print(ser1) #this returns:
#USA        1
#China      2
#Russia     3
#England    4
#dtype: int64
# we can get tthe value of indexes like dictionaries:
print(ser1["USA"]) #this returns 1
print(ser1["England"]) #this returns the value of this index like keys in the dictionaries and returns as 4
print(".......................")
ser2=pd.Series([1,2,6,8],["USA","China","Turkey","Italy"])
print(ser1+ser2) #this sums the values if the two series has the same label names, if not this returns NaN for those which are not in both series
#China      4.0
#England    NaN
#Italy      NaN
#Russia     NaN
#Turkey     NaN
#USA        2.0 # the integers will be represeneted as floats
#dtype: float64


""" 2: DATAFRAMES:
    Dataframes are  main objects we work with pandas
"""
from numpy.random import randn
#np.random.seed(0) makes the random numbers predictable,With the seed reset (every time), the same set of numbers will appear every time.
np.random.seed(101)
print(np.random.randn(10))
print("..................................")
print(np.random.randn(5,4)) #here we create 5 rows with 4 columns with randomly distributed
# when we want to show them in a dataframe in a tabular form we can use pd.DataFrame() function
df=pd.DataFrame(randn(5,4),["A","B","C","D","E"],["W","X","Y","Z"])
print(df)
#Actually pandas dataframe is a bunch of series
# Like dictinaries we can use dataframe[column name] to reach the data in the column

"""Selecting Columns: """
print(df["W"]) # this returns a series, type(df) returns dataframe while df[columname] returns a series
print(df["X"])
# we can also get more columns via df[[column name,column name]], we should use one more [] that includes columns
print(df[["X","Y","Z"]]) # this returns 3 columns as a dataframe type

# we can create new columns via gving new data for each row or making mathematical operation with existing columns
df["New"]=df["X"]+df["Z"] #here we create a column by using previous columns
print(df["New"])
print(df)
df["New2"]=[0.232,0.3434,0.34343,0.3434,0.43443] #here we create a new column with the data we assign
print(df)
"""
drop(labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise') method of pandas.core.frame.DataFrame instance
    Drop specified labels from rows or columns. 
    Remove rows or columns by specifying label names and corresponding axis
    labels : single label or list-like
        Index or column labels to drop.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Whether to drop labels from the index (0 or 'index') or
        columns (1 or 'columns')."""
print(df)
df.drop("New2",axis=1,inplace=True) # we have specify inplace=True in order to drop column in the dataframe.
print(df)
# we can also drop rows by selecting axis=0
df.drop("A",axis=0,inplace=True)
print(df)

"""Selecting Rows:There are two ways to select rows:df.loc[]takes label as an argument while df.iloc[index] gets index
Both methods returns the same output"""
x=df.loc["B"]
print(x)
y=df.iloc[0]
print(y)
#we can also get the row sets like the way in numpy
z=df.loc["B":"D","W":"Y"] #this retuns the rows between "B" and "D" and columns "W" and "Y"; the names of rows and columns are inclusive
print(z)
"""Conditional Selection: """
booldf=df >1 #This returns True or False in each rows and column combinations in the dataframe
print(booldf)
print(df[booldf]) #this returns values that compy with condition and gives Nan for false ones
print(df)
print(df[df["W"]> 0]) #this return only the row or rows where df["W"] > 0
print(df[df["W"] > 0]["W"]) #This returns only the conditions "W" column
print(df[df["W"] > 0][["W","Y"]]) #this returns the conditions "W" and "Y" columns
print(df[df["W"]<0]) #returns the rows where df["W"] < 0
print("....................................")
# we can make more than one condition:
"""we can not use or and words, we have to use their symbols to use them """
print(df[(df["W"] < 0) & (df["Y"] > 0)]) #this returns the rows where df["W"] < 0 and df["Y"] >0
print(df[(df["W"] < 9) | (df["Y"] >0)]) #this returns the rows where df["W"] < 0 or df["Y"] >0

""" in order to reset index back to the default we use df.reset_index() """
print(df.reset_index()) # we will get default index system for rows instead of what we give
print(df) # if we want to get back default index we have to specify inplace=True as df.reset_index(inplace=True)
#the old index will be a columns after that
newindex="CA NY LA OR".split() # here we create new indices with .split()
df["States"]=newindex
print(df)

"""Multi index and Index Hierarchy:"""
outside="G1 G1 G1 G2 G2 G2".split()
inside=[1,2,3,1,2,3]
hier_index=list(zip(outside,inside))
#pd.MultiIndex() created multi indices
hier_index=pd.MultiIndex.from_tuples(hier_index) 
df=pd.DataFrame(randn(6,2),hier_index,["A","B"]) # here we cerated data frame with two main indices as G1 and G2
print(df)
#             A         B
#G1 1 -0.755325 -0.346419
#   2  0.147027 -0.479448
#   3  0.558769  1.024810
#G2 1 -0.925874  1.862864
#   2 -1.133817  0.610478
#   3  0.386030  2.084019
print(df.loc["G1"])
#          A         B
#1 -0.755325 -0.346419
#2  0.147027 -0.479448
#3  0.558769  1.024810
print(df.loc["G1"].loc[1]) # we can get detail indexing via using .loc[] many times 
#A   -0.755325
#B   -0.346419
print(df["A"])
df.index.names=["Groups","Number"] #"here we give names for indices"
print(df.loc["G2"].loc[2].loc["B"])

#df.xs() returns cross section of columns and rows: we can different information from different section in the dataframe at the same time:
print(df)
print(df.xs(1,level="Number")) #this returns all the information within different groups:
#               A         B
#Groups                    
#G1     -0.755325 -0.346419
#G2     -0.925874  1.862864


"""Missing Data:
    1.We use df.dropna() to drop the missing data from the dataframe
    dropna(axis=0, how='any', thresh=None, subset=None, inplace=False) method of pandas.core.frame.DataFrame instance
    Remove missing values.
    axis : {0 or 'index', 1 or 'columns'}, default 0
        Determine if rows or columns which contain missing values are
        removed."""
dct={"A":[1,2,np.nan], "B":[12,23,np.nan],"C":[12,32,34]}
df=pd.DataFrame(dct)
print(df)
#     A     B   C
#0  1.0   NaN  12
#1  2.0  23.0  32
#2  NaN   NaN  34
df.dropna() #this returns only the rows without missing value
print(df)
#  A     B   C
#0  1.0  12.0  12
#1  2.0  23.0  32
#if we want to specify columns we need write axis:1 because the default is axis:0 which is for rows
df.dropna(axis=1)
print(df)
#    C
#0  12
#1  32
#2  34
# we can specify for dropping missing value by filling thresh=int
df.dropna(axis=1,thresh=2) # here we command that just drop columns those which have two or more missing values
print(df)
"""df.fillna() enables us to fill the mising values in the dataframe

"""
df.fillna(value="Fill Value")
print(df) #             A           B   C
#0           1          12  12
#1           2          23  32
#2  Fill Value  Fill Value  34
#we can also specify which column we want to fill the missing values
df["A"].fillna(value=df["A"].mean(),inplace=True)
print(df) # here we fill the misisng value in the A column with the mean of the column as follows:
#     A     B   C
#0  1.0  12.0  12
#1  2.0  23.0  32
#2  1.5   NaN  34
df["B"].fillna(value=df.mean(),inplace=True)
print(df)


"""df.groupby(): allows us to group together rows and perform aggregate function on them,this means it takes many values and create one value
    Aggreagte functions like sum of values in a ine single value
    groupby(by=None, axis=0, level=None, as_index: bool = True, sort: bool = True, group_keys: bool = True, squeeze: bool = False, observed: bool = False) -> 'groupby_generic.DataFrameGroupBy' method of pandas.core.frame.DataFrame instance
    Group DataFrame using a mapper or by a Series of columns.
    df.groupby(the column we want to groupby)+aggreagte methods"""
data={'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
     'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
      'Sales':[200,120,340,124,243,350]}
df=pd.DataFrame(data)
#  Company   Person  Sales
#0    GOOG      Sam    200
#1    GOOG  Charlie    120
#2    MSFT      Amy    340
#3    MSFT  Vanessa    124
#4      FB     Carl    243
#5      FB    Sarah    350
com=df.groupby("Company") # here we create a groupby object: the companies in the same name from different rows have been grouped by one group
print(com) #<pandas.core.groupby.generic.DataFrameGroupBy object at 0x00000283F7938F48>
# now we can use aggregate function after groupering:
print(com.mean()) # here it returns average sales by company because it ignores non numeric columns for mean
print(dir(com))
print(com.sum()) #this returns sum of sales for each company
print(com.min()) # minimum sale
print(com.max()) # max sales
print(com.std()) # standart deviation
print(com.size()) 
print(com.median())
print(com.sum().loc["FB"]) #this returns only the sum of sales from "FB" company
print(com.count()) # returns number of items in the dataframe
#Company               
#FB            2      2
#GOOG          2      2
#MSFT          2      2
print(com.describe()) #"""This returns all of the statistical information of all groups:"""
#        count   mean         std    min     25%    50%     75%    max
#Company                                                              
#FB        2.0  296.5   75.660426  243.0  269.75  296.5  323.25  350.0
#GOOG      2.0  160.0   56.568542  120.0  140.00  160.0  180.00  200.0
#MSFT      2.0  232.0  152.735065  124.0  178.00  232.0  286.00  340.0
print(com.describe().transpose()) # we can exchange the place of columns and rows:
#Company              FB        GOOG        MSFT
#Sales count    2.000000    2.000000    2.000000
#      mean   296.500000  160.000000  232.000000
#      std     75.660426   56.568542  152.735065
#      min    243.000000  120.000000  124.000000
#      25%    269.750000  140.000000  178.000000
#      50%    296.500000  160.000000  232.000000
#      75%    323.250000  180.000000  286.000000
#      max    350.000000  200.000000  340.000000
print(com.describe().loc["FB"]) #this returns statistical information only of FB
#Sales  count      2.000000
 #      mean     296.500000
 #      std       75.660426
 #      min      243.000000
 #      25%      269.750000
 #      50%      296.500000
 #     75%      323.250000
 #      max      350.000000
#Name: FB, dtype: float64




""" Merging,Joining and Concatenating Data Frames: """

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                            'B': ['B0', 'B1', 'B2', 'B3'],
                           'C': ['C0', 'C1', 'C2', 'C3'],
                           'D': ['D0', 'D1', 'D2', 'D3']},
                          index=[0, 1, 2, 3])
df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                            'B': ['B4', 'B5', 'B6', 'B7'],
                            'C': ['C4', 'C5', 'C6', 'C7'],
                            'D': ['D4', 'D5', 'D6', 'D7']},
                             index=[4, 5, 6, 7]) 
df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                           'B': ['B8', 'B9', 'B10', 'B11'],
                            'C': ['C8', 'C9', 'C10', 'C11'],
                            'D': ['D8', 'D9', 'D10', 'D11']},
                            index=[8, 9, 10, 11])
print(df1)
print(df2)
print(df3)

"""Concatenating glues together the data frames:
    pd.concat([list of dataframes])
    pd.concat(objs: Union[Iterable[Union[ForwardRef('DataFrame'), ForwardRef('Series')]], Mapping[Union[Hashable, 
    NoneType], Union[ForwardRef('DataFrame'), ForwardRef('Series')]]], axis=0, join='outer', ignore_index: bool = False,
    keys=None, levels=None, names=None, verify_integrity: bool = False, sort: bool = False, 
    copy: bool = True) -> Union[ForwardRef('DataFrame'), ForwardRef('Series')]
    Concatenate pandas objects along a particular axis with optional set logic
    along the other axes.
    But dimensions of the data frame should match with each other"""
con=pd.concat([df1,df2,df3]) # defaul axis=0 it means that this function will join rows together
print(con)
# A    B    C    D
#0    A0   B0   C0   D0
#1    A1   B1   C1   D1
#2    A2   B2   C2   D2
#3    A3   B3   C3   D3
#4    A4   B4   C4   D4
#5    A5   B5   C5   D5
#6    A6   B6   C6   D6
#7    A7   B7   C7   D7
#8    A8   B8   C8   D8
#9    A9   B9   C9   D9
#10  A10  B10  C10  D10
con1=pd.concat([df1,df2,df2], axis=1) #this returns with missing information
print(con1)
#A    B    C    D    A    B    C    D    A    B    C    D
#0   A0   B0   C0   D0  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
#1   A1   B1   C1   D1  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
#2   A2   B2   C2   D2  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
#3   A3   B3   C3   D3  NaN  NaN  NaN  NaN  NaN  NaN  NaN  NaN
#4  NaN  NaN  NaN  NaN   A4   B4   C4   D4   A4   B4   C4   D4
#5  NaN  NaN  NaN  NaN   A5   B5   C5   D5   A5   B5   C5   D5
#6  NaN  NaN  NaN  NaN   A6   B6   C6   D6   A6   B6   C6   D6
#7  NaN  NaN  NaN  NaN   A7   B7   C7   D7   A7   B7   C7   D7



"""The **merge** function allows you to merge DataFrames together using a similar logic as merging SQL Tables together.
The difference between merge and concatenate is that marge absorb the same columns and show them in a one column
merge(left, right, how: str = 'inner', on=None, left_on=None, right_on=None, left_index: bool = False, right_index: bool = False, sort: bool = False, suffixes=('_x', '_y'), copy: bool = True, indicator: bool = False, validate=None) -> 'DataFrame'
    Merge DataFrame or named Series objects with a database-style join.
    
    The join is done on columns or indexes. If joining columns on
    columns, the DataFrame indexes *will be ignored*. Otherwise if joining indexes
    on indexes or indexes on a column or columns, the index will be passed on.
    Parameters
    ----------
    left : DataFrame
    right : DataFrame or named Series
        Object to merge with.
    how : {'left', 'right', 'outer', 'inner'}, default 'inner'
        Type of merge to be performed.
    
        * left: use only keys from left frame, similar to a SQL left outer join;
          preserve key order.
        * right: use only keys from right frame, similar to a SQL right outer join;
          preserve key order.
        * outer: use union of keys from both frames, similar to a SQL full outer"""


left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                        'A': ['A0', 'A1', 'A2', 'A3'],
                         'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                            'C': ['C0', 'C1', 'C2', 'C3'],
                             'D': ['D0', 'D1', 'D2', 'D3']}) 
new=pd.merge(left,right,on="key") # on specify which column is the starting for merging
print(left)
#  key   A   B
#0  K0  A0  B0
#1  K1  A1  B1
#2  K2  A2  B2
#3  K3  A3  B3
print(right)
#  key   C   D
#0  K0  C0  D0
#1  K1  C1  D1
#2  K2  C2  D2
#3  K3  C3  D3
print(new)
#  key   A   B   C   D
#0  K0  A0  B0  C0  D0
#1  K1  A1  B1  C1  D1
#2  K2  A2  B2  C2  D2
#3  K3  A3  B3  C3  D3


""" Joining:
Joining is a convenient method for combining the columns of two potentially differently-indexed DataFrames 
into a single result DataFrame.
This method is like pd.merger() function, but it uses indices or rows nstead of columns:
    join(other, on=None, how='left', lsuffix='', rsuffix='', sort=False) -> 'DataFrame' method of pandas.core.frame.DataFrame instance
    Join columns of another DataFrame.
    
    Join columns with `other` DataFrame either on index or on a key
    column. Efficiently join multiple DataFrame objects by index at once by
    passing a list.
    """
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                        'B': ['B0', 'B1', 'B2']},
                          index=['K0', 'K1', 'K2'])
print(left)
#   A   B
#K0  A0  B0
#K1  A1  B1
#K2  A2  B2
    
right = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                       'D': ['D0', 'D2', 'D3']},
                       index=['K0', 'K2', 'K3'])
print(right)
#  C   D
#K0  C0  D0
#K2  C2  D2
#K3  C3  D3
print(left.join(right))
#    A   B    C    D
#K0  A0  B0   C0   D0
#K1  A1  B1  NaN  NaN
#K2  A2  B2   C2   D2
print(left.join(right,how="outer"))
# A    B    C    D
#K0   A0   B0   C0   D0
#K1   A1   B1  NaN  NaN
#K2   A2   B2   C2   D2
#K3  NaN  NaN   C3   D3
print(pd.concat([right,left]))
#      C    D    A    B
#K0   C0   D0  NaN  NaN
#K2   C2   D2  NaN  NaN
#K3   C3   D3  NaN  NaN
#K0  NaN  NaN   A0   B0
#K1  NaN  NaN   A1   B1
#K2  NaN  NaN   A2   B2




"""Operations in Pandas."""

df = pd.DataFrame({'col1':[1,2,3,4],'col2':[444,555,666,444],'col3':['abc','def','ghi','xyz']}) 
print(df)

# df.unique() finds unique value in the data frame:
print(df["col2"].unique()) #this returns a numpy array that has only the uniques values in the column
#[444 555 666]

#df.nunique() returns number of unique values in the column
print(df["col2"].nunique()) # returns 3

#df.value_counts() returns a table of unique values and how many they are
print(df["col2"].value_counts())
#444    2
#555    1
#666    1
#Name: col2, dtype: int64

#df.apply() method enable us to use any function for the dataframe
def times2(x):
    return x * 2
df=df.apply(times2) #here we apply times() function and all the values in the data frame is multiplied by 2 including strings
print(df)
#   col1  col2    col3
#0     2   888  abcabc
#1     4  1110  defdef
#2     6  1332  ghighi
#3     8   888  xyzxyz
# we can also implement this method for separate columns:
x=df["col2"].apply(times2)
print(x)
#0    1776
#1    2220
#2    2664
#3    1776
# df.apply() method accepts also builtin functions
print(df.apply(len)) # returns the length of each columns
#col1    4
#col2    4
#col3    4
#dtype: int64
#df.apply() can also accept lambda functions
x=df.apply(lambda x: x *3)
print(x)
#   col1  col2                col3
#0     6  2664  abcabcabcabcabcabc
#1    12  3330  defdefdefdefdefdef
#2    18  3996  ghighighighighighi
#3    24  2664  xyzxyzxyzxyzxyzxyz



"""Sorting Data Frames:
    df.sort_values(column or row name)
    sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last', ignore_index=False) method of pandas.core.frame.DataFrame instance
    Sort by the values along either axis.
    
    Parameters
    ----------
            by : str or list of str
                Name or list of names to sort by.
    
                - if `axis` is 0 or `'index'` then `by` may contain index
                  levels and/or column labels.
                - if `axis` is 1 or `'columns'` then `by` may contain column
                  levels and/or index labels."""
print(df)
print(df.sort_values("col2")) # here we sort values in the column 2
# The index of the column do not change, so we do not lose information, index stays attached its previous row or column



"""df.isnull() enables us to find null values in the data frame and return True or False
If there is a null value, this returns true for that value"""
print(df.isnull())
#col1   col2   col3
#0  False  False  False
#1  False  False  False
#2  False  False  False
#3  False  False  False
data = {'A':['foo','foo','foo','bar','bar','bar'],
         'B':['one','one','two','two','one','one'],
          'C':['x','y','x','y','x','y'],
          'D':[1,3,2,5,4,1]}
df=pd.DataFrame(data)
print(df)
#   A    B  C  D
#0  foo  one  x  1
#1  foo  one  y  3
#2  foo  two  x  2
#3  bar  two  y  5
#4  bar  one  x  4
#5  bar  one  y  1
print(df.groupby("A"))


""" data Input and Output in Pandas:"""
df=pd.read_excel("Excel_Sample.xlsx")
print(df) # each sheet in excel is a dataframe in pandas

df.to_excel("MySample.xlsx") # here we create or save existing file into a new file 
df.to_csv("MySample2.csv") # here we transfomred excel file into a csv file via pandas
df=pd.read_csv("MySample2.csv")
print(df)


"""pd.read_html(url) reads the html files from the url we specify """
data = pd.read_html('http://www.fdic.gov/bank/individual/failed/banklist.html')
print(data) # this return a list of data from the html file
print(data[0].head())

"""
    every SQL has its own python library to work with like MySQL, PostgreSQL"""
from sqlalchemy import create_engine # this creates a SQL engine in the memory
#engine=create_engine("sqlite:///:memory")
#df.to_sql("mysql_table",engine)# the first one is the name of SQL file while the second is the name engine We will work with








        
    


    






