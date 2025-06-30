# Barkemeyer_Python_Portfolio
This is the portfoltio of python code that I learned in BISC 450C.
## Using Jupiter Notebooks

```python
# To get started, import packages, libraries needed for analysis:

%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = "darkgrid")
```


```python
df = pd.read_csv('/home/student/Desktop/classroom/myfiles/notebooks/fortune500.csv')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Rank</th>
      <th>Company</th>
      <th>Revenue (in millions)</th>
      <th>Profit (in millions)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>25495</td>
      <td>2005</td>
      <td>496</td>
      <td>Wm. Wrigley Jr.</td>
      <td>3648.6</td>
      <td>493</td>
    </tr>
    <tr>
      <td>25496</td>
      <td>2005</td>
      <td>497</td>
      <td>Peabody Energy</td>
      <td>3631.6</td>
      <td>175.4</td>
    </tr>
    <tr>
      <td>25497</td>
      <td>2005</td>
      <td>498</td>
      <td>Wendy's International</td>
      <td>3630.4</td>
      <td>57.8</td>
    </tr>
    <tr>
      <td>25498</td>
      <td>2005</td>
      <td>499</td>
      <td>Kindred Healthcare</td>
      <td>3616.6</td>
      <td>70.6</td>
    </tr>
    <tr>
      <td>25499</td>
      <td>2005</td>
      <td>500</td>
      <td>Cincinnati Financial</td>
      <td>3614.0</td>
      <td>584</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns = ['year', 'rank', 'company', 'revenue', 'profit']
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1955</td>
      <td>1</td>
      <td>General Motors</td>
      <td>9823.5</td>
      <td>806</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1955</td>
      <td>2</td>
      <td>Exxon Mobil</td>
      <td>5661.4</td>
      <td>584.8</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1955</td>
      <td>3</td>
      <td>U.S. Steel</td>
      <td>3250.4</td>
      <td>195.4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1955</td>
      <td>4</td>
      <td>General Electric</td>
      <td>2959.1</td>
      <td>212.6</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1955</td>
      <td>5</td>
      <td>Esmark</td>
      <td>2510.8</td>
      <td>19.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df)
```




    25500




```python
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit      object
    dtype: object




```python
# To view profits that contain anything but 0-9:

non_numeric_profits = df.profit.str.contains('[^0-9.-]')
df.loc[non_numeric_profits].head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>rank</th>
      <th>company</th>
      <th>revenue</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>228</td>
      <td>1955</td>
      <td>229</td>
      <td>Norton</td>
      <td>135.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>290</td>
      <td>1955</td>
      <td>291</td>
      <td>Schlitz Brewing</td>
      <td>100.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>294</td>
      <td>1955</td>
      <td>295</td>
      <td>Pacific Vegetable Oil</td>
      <td>97.9</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>296</td>
      <td>1955</td>
      <td>297</td>
      <td>Liebmann Breweries</td>
      <td>96.0</td>
      <td>N.A.</td>
    </tr>
    <tr>
      <td>352</td>
      <td>1955</td>
      <td>353</td>
      <td>Minneapolis-Moline</td>
      <td>77.4</td>
      <td>N.A.</td>
    </tr>
  </tbody>
</table>
</div>




```python
# To see what makes up non-numeric entries:

set(df.profit[non_numeric_profits])
```




    {'N.A.'}




```python
# To view how many entries are 'N.A':

len(df.profit[non_numeric_profits])
```




    369




```python
# To create a histogram using data frame year as x-axis and non_numeric_profits as count value:
    
bin_sizes, _, _ = plt.hist(df.year[non_numeric_profits], bins = range(1955, 2006))
```


![png](output_11_0.png)



```python
# To remove the NAs (369):
df = df.loc[-non_numeric_profits]
df.profit = df.profit.apply(pd.to_numeric)
```


```python
len(df)
```




    25131




```python
# To see that profit is now a float data type instead of object:
df.dtypes
```




    year         int64
    rank         int64
    company     object
    revenue    float64
    profit     float64
    dtype: object




```python
# Let's do some plotting:
group_by_year = df.loc[:, ['year', 'revenue', 'profit']].groupby('year')
avgs = group_by_year.mean()
x = avgs.index
y1 = avgs.profit
def plot(x, y, ax, title, y_label):
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.plot(x,y)
    ax.margins(x=0, y=0)
```


```python
fig, ax = plt.subplots()
plot(x, y1, ax, 'Increase in mean Fortune 500 company profits from 1955-2005', 'Profit (millions)')
```


![png](output_16_0.png)



```python
# To show revenues of companies during the same period:
y2 = avgs.revenue
fig, ax = plt.subplots()
plot(x, y2, ax, 'Increase in mean Fortune 500 company revenues from 1955-2005', 'Revenue (millions)')
```


![png](output_17_0.png)



```python
def plot_with_std(x, y, stds, ax, title, y_label):
    ax.fill_between(x, y - stds, y + stds, alpha = 0.2)
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_ylabel(y_label)
fig, (ax1, ax2) = plt.subplots(ncols = 2) 
title = 'Increase in mean and std fortune 500 company %s from 1955 to 2005'
stds1 = group_by_year.std().profit.values
stds2 = group_by_year.std().revenue.values
plot_with_std(x, y1.values, stds1, ax1, title % 'profits', 'Profit (millions)')
plot_with_std(x, y2.values, stds2, ax2, title % 'revenues', 'Revenue (millions)')
fig.set_size_inches(14, 4)
fig.tight_layout()
```


![png](output_18_0.png)


## Python Fundamentals
```python

# Any python interpreter can be used a a calculator:

3 +5 *4
```




    23




```python
# To save a value to a variable example:

weight_kg = 60
```


```python
print(weight_kg)
```

    60



```python
# Some python rules:

# Weight0 = valid
# 0weight = invalid
# * Note: A variable name cannot start with a number.
# Also, variable names are case sensitive.
```


```python
# Types of Data

# There are 3 common types of data:
# * Integer numbers
# * Floating point numbers (decimal)
# * Strings - not a number value, but may contain a number (i.e. ID)
```


```python
# Floating point number example:
weight_kg = 60.3
```


```python
# String examples:

patient_name = "John Smith"

patient_id = '001'
```


```python
# To use variables in python example:

weight_lb = 2.2*weight_kg
```


```python
print(weight_lb)
```

    132.66



```python
# Let's add a prefix to our patient id:

patient_id = 'inflam_' + patient_id
```


```python
print(patient_id)
```

    inflam_001



```python
# To combine print statements:

print(patient_id, 'weight in kilograms:', weight_kg)
```

    inflam_001 weight in kilograms: 60.3



```python
# To call a function inside another function:

# floating number example:

print(type(60.3))
```

    <class 'float'>



```python
# string example:

print(type(patient_id))
```

    <class 'str'>



```python
# We can also do calculations inside the print function:

print('weight in lbs:', 2.2*weight_kg)
```

    weight in lbs: 132.66



```python
print(weight_kg)
# *Note: We did not save over the variable.
```

    60.3



```python
# To save over a variable:

weight_kg = 65.0
print('weight in kilograms is now:', weight_kg)
```

    weight in kilograms is now: 65.0



```python

```

## Analyzing Patient Data

In this analysis, we looked at inflammation data for multiple patients.
```python
# Numerical Python
import numpy
```


```python
numpy.loadtxt(fname = 'Python_Data/inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
# To make data variable:
data = numpy.loadtxt(fname = 'Python_Data/inflammation-01.csv', delimiter = ',')
print(data)
print(type(data))

```

    [[0. 0. 1. ... 3. 0. 0.]
     [0. 1. 2. ... 1. 0. 1.]
     [0. 1. 1. ... 2. 1. 1.]
     ...
     [0. 1. 1. ... 1. 1. 1.]
     [0. 0. 0. ... 0. 2. 0.]
     [0. 0. 1. ... 1. 1. 0.]]
    <class 'numpy.ndarray'>



```python
# To see size/shape of array:
print(data.shape)
print('first value in data:', data[0,0])

```

    (60, 40)
    first value in data: 0.0



```python
# first place in a data array  is 0,0 (not 1,1)
print('middle value in data:', data[29,19])

```

    middle value in data: 16.0



```python
print (data[0:4, 0:10])
# this means start at 0 and go up to 4 (do not include 4) and go up to 10 for columns (don’t include 10)

```

    [[0. 0. 1. 3. 1. 2. 4. 7. 8. 3.]
     [0. 1. 2. 1. 2. 1. 3. 2. 2. 6.]
     [0. 1. 1. 3. 3. 2. 6. 2. 5. 9.]
     [0. 0. 2. 0. 4. 2. 2. 1. 6. 7.]]



```python
print(data[5:10, 0:10])
```

    [[0. 0. 1. 2. 2. 4. 2. 1. 6. 4.]
     [0. 0. 2. 2. 4. 2. 2. 5. 5. 8.]
     [0. 0. 1. 2. 3. 1. 2. 3. 5. 3.]
     [0. 0. 0. 3. 1. 5. 6. 5. 5. 8.]
     [0. 1. 1. 2. 1. 3. 5. 3. 5. 8.]]



```python
small = data[:3, 36:]
print ('small is:')
print(small)

```

    small is:
    [[2. 3. 0. 0.]
     [1. 1. 0. 1.]
     [2. 2. 1. 1.]]



```python
# Let’s use a numpy function
print(numpy.mean(data))

```

    6.14875



```python
# To make a merged function:
maxval, minval, stdval = numpy.amax(data), numpy.amin(data), numpy.std(data)
print(maxval)
print(minval)
print(stdval)
print('maximum inflammation:', maxval)
print('minimum inflammation:', minval)
print('standard deviation:', stdval)

```

    20.0
    0.0
    4.613833197118566
    maximum inflammation: 20.0
    minimum inflammation: 0.0
    standard deviation: 4.613833197118566



```python
# Sometimes we want to look at variation in statistical values, such as maximum inflammation per patient or average from day one.
patient_0 = data[0, :]
# 0 on the first axis (rows), everything on the second (columns)
print('maximum inflammation for patient 0:', numpy.amax(patient_0))
```

    maximum inflammation for patient 0: 18.0



```python
print('maximum inflammation for patient 2:', numpy.amax(data [2, :]))
```

    maximum inflammation for patient 2: 19.0



```python
print(numpy.mean(data, axis = 0))
```

    [ 0.          0.45        1.11666667  1.75        2.43333333  3.15
      3.8         3.88333333  5.23333333  5.51666667  5.95        5.9
      8.35        7.73333333  8.36666667  9.5         9.58333333 10.63333333
     11.56666667 12.35       13.25       11.96666667 11.03333333 10.16666667
     10.          8.66666667  9.15        7.25        7.33333333  6.58333333
      6.06666667  5.95        5.11666667  3.6         3.3         3.56666667
      2.48333333  1.5         1.13333333  0.56666667]



```python
# average inflammation of all patients for a given day (column avg)
print(numpy.mean(data, axis = 0).shape)

```

    (40,)



```python
# to see the average inflammation per person (row avg):
print(numpy.mean(data, axis =1))

```

    [5.45  5.425 6.1   5.9   5.55  6.225 5.975 6.65  6.625 6.525 6.775 5.8
     6.225 5.75  5.225 6.3   6.55  5.7   5.85  6.55  5.775 5.825 6.175 6.1
     5.8   6.425 6.05  6.025 6.175 6.55  6.175 6.35  6.725 6.125 7.075 5.725
     5.925 6.15  6.075 5.75  5.975 5.725 6.3   5.9   6.75  5.925 7.225 6.15
     5.95  6.275 5.7   6.1   6.825 5.975 6.725 5.7   6.25  6.4   7.05  5.9  ]



```python
# Import plotting library
import matplotlib.pyplot

```


```python
# To make image
image = matplotlib.pyplot.imshow(data)

```


![png](output_16_0.png)



```python
# To show image
matplotlib.pyplot.show()

```


```python
# you can see that inflammation peaks around day 19 for all patients

# After taking medication, inflammation appears to drop to 0 after 3 weeks.
```


```python
# Average inflammation over time:
ave_inflammation  = numpy.mean(data, axis =0)
ave_plot = matplotlib.pyplot.plot(ave_inflammation)
matplotlib.pyplot.show()


```


![png](output_19_0.png)



```python
# We can see time (weeks) on x-axis and inflammation on y-axis and inflammation peaks around week 20.
```


```python
# to plot the max
max_plot = matplotlib.pyplot.plot(numpy.amax(data, axis =0))
matplotlib.pyplot.show()

```


![png](output_21_0.png)



```python
# to plot the min
min_plot = matplotlib.pyplot.plot(numpy.amin(data, axis = 0))
matplotlib.pyplot.show()

```


![png](output_22_0.png)



```python
import numpy
# to group plots together
fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))
axes1 = fig.add_subplot(1, 3, 1)
axes2 = fig.add_subplot(1, 3, 2)
axes3 = fig.add_subplot(1, 3, 3)

axes1.set_ylabel('average')
axes1.plot(numpy.mean(data, axis = 0))

axes2.set_ylabel('max')
axes2.plot(numpy.amax(data, axis = 0))

axes3.set_ylabel('min')
axes3.plot(numpy.amin(data, axis = 0))

# This will create one plot with 3 "pictures." Axes1 (average) is in the first position(1,3,1), Axes2 (maximum),
#second, etc.

fig.tight_layout()

matplotlib.pyplot.savefig('inflammation.png')
matplotlib.pyplot.show()
```


![png](output_23_0.png)



```python

```
## Storing Values in Lists

```python
# An example of a list:

odds = [1,3,5,7]
```


```python
print('odds are:', odds)
```

    odds are: [1, 3, 5, 7]



```python
print('first element:', odds[0])
```

    first element: 1



```python
print('last element:', odds[3])
```

    last element: 7



```python
# - means counting backwards:

print('"-1" element:', odds[-1])
print('"-2" element:', odds[-2])
print('"-3" element:', odds[-3])
```

    "-1" element: 7
    "-2" element: 5
    "-3" element: 3



```python
names= ['Curie', 'Darwig', 'Turing']
# Note typeo in Darwin's name
print('names is originally:', names)
```

    names is originally: ['Curie', 'Darwig', 'Turing']



```python
# To correct the name:
names[1] = 'Darwin'
```


```python
print('final value of names:', names)
```

    final value of names: ['Curie', 'Darwin', 'Turing']



```python
# name = 'Darwin'
# name[0] = 'd'
```


```python
# To add 11 to our Odds List:
odds.append(11)
print('odds after adding a value:', odds)
```

    odds after adding a value: [1, 3, 5, 7, 11]



```python
# To remove 1 from the list:
removed_element = odds.pop(0)
print('odds after removing the first element:', odds)
print('removed_element:', removed_element)
```

    odds after removing the first element: [3, 5, 7, 11]
    removed_element: 1



```python
# To reverse the list:
odds.reverse()
print('odds after reversing:', odds)
```

    odds after reversing: [11, 7, 5, 3]



```python
# Adds 2 to both lists (we don't want this):
odds = [3,5,7]
primes = odds
primes.append(2)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7, 2]
    odds: [3, 5, 7, 2]



```python
# To avoid appending something to both lists, use the list function:
odds = [3,5,7]
primes = list(odds)
print('primes:', primes)
print('odds:', odds)
```

    primes: [3, 5, 7]
    odds: [3, 5, 7]



```python
# To slice a list or string:

binomial_name = "Drosophila melanogaster"
group = binomial_name[0:10]
print('group:', group)

species = binomial_name[11:23]
print('species:', species)

chromosomes = ['X', 'Y', '2', '3', '4']
autosomes = chromosomes[2:5]
print('autosomes:', autosomes)

last = chromosomes[-1]
print('last:', last)
```

    group: Drosophila
    species: melanogaster
    autosomes: ['2', '3', '4']
    last: 4



```python
date = 'Monday 4 January 2025'
day = date[0:6]
# This means the first 6 characters
print('Using 0 to begin range:', day)
day = date[:6]
print('Omitting beginning indes:', day)
```

    Using 0 to begin range: Monday
    Omitting beginning indes: Monday



```python
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
sond = months[8:12]
print('With known last position:', sond)

# what if we don't know the last position:

sond = months[8:len(months)]
print('Using len() to get last entry:', sond)
sond = months[8: ]

print('Omitting ending index:', sond)
```

    With known last position: ['sep', 'oct', 'nov', 'dec']
    Using len() to get last entry: ['sep', 'oct', 'nov', 'dec']
    Omitting ending index: ['sep', 'oct', 'nov', 'dec']



```python

```
## Using Loops

```python

odds = [1,3,5,7]
```


```python
# To print each one by one, type a line of code for each:

print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])

```

    1
    3
    5
    7



```python


odds = [1,3,5]
print(odds[0])
print(odds[1])
print(odds[2])
print(odds[3])
# we get an error b/c 3 isn't there
```

    1
    3
    5



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-3-8947594eb0f3> in <module>
          3 print(odds[1])
          4 print(odds[2])
    ----> 5 print(odds[3])
          6 # we get an error b/c 3 isn't there


    IndexError: list index out of range



```python

# Or we can code for a loop

odds = [1,3,5,7]

for num in odds:
    print(num)
```

    1
    3
    5
    7



```python
# To go back and add more to list and loop still prints:

odds = [1,3,5,7,9,11,13,15,17,19]

for num in odds:
    print(num)
```

    1
    3
    5
    7
    9
    11
    13
    15
    17
    19



```python
# Another example:

length = 0
names = ['Curie', 'Darwin', 'Turing']
for value in names:
    length = length + 1
print('There are', length, 'names in the list.')
```

    There are 3 names in the list.



```python
# Do not reuse a variable, b/c it will write over the original:
name = "Rosalind"
for name in ['Curie', 'Darwin', 'Turing']:
    print(name)
print('after the loop, name is', name)
```

    Curie
    Darwin
    Turing
    after the loop, name is Turing



```python
# A common function in Python:

print(len([0,1,2,3]))
```

    4



```python
name = ['Curie', 'Darwin', 'Turing']

print(len(name))
```

    3



```python

```

## Using Multiple Files

```python
import glob
```


```python
print(glob.glob('Python_Data/inflammation*.csv'))
```

    ['Python_Data/inflammation-10.csv', 'Python_Data/inflammation-09.csv', 'Python_Data/inflammation-11.csv', 'Python_Data/inflammation-06.csv', 'Python_Data/inflammation-05.csv', 'Python_Data/inflammation-08.csv', 'Python_Data/inflammation-01.csv', 'Python_Data/inflammation-07.csv', 'Python_Data/inflammation-04.csv', 'Python_Data/inflammation-03.csv', 'Python_Data/inflammation-02.csv', 'Python_Data/inflammation-12.csv']



```python
import glob
import numpy
import matplotlib.pyplot

# To create a variable of sorted inflammation files, then, just take first 3 and save them as filenames
filenames = sorted(glob.glob('Python_Data/inflammation*.csv'))
filenames = filenames[0:3]

for filename in filenames:
    print(filename)
    
    data = numpy.loadtxt(fname=filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize = (10.0, 3.0))
    
    axes1 = fig.add_subplot(1,3,1)
    axes2 = fig.add_subplot(1,3,2)
    axes3 = fig.add_subplot(1,3,3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis = 0))
    axes2.set_ylabel('max')
    axes2.plot(numpy.amax(data, axis = 0))
    axes3.set_ylabel('min')
    axes3.plot(numpy.amin(data, axis = 0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()
```

    Python_Data/inflammation-01.csv



![png](output_2_1.png)


    Python_Data/inflammation-02.csv



![png](output_2_3.png)


    Python_Data/inflammation-03.csv



![png](output_2_5.png)



```python

```
## Making Choices

```python
# Making Choices
# What if we want to write a script to decide something:

# First, create variable number, then 'if', tells what to print
num = 37
if num > 100: 
    print('greater')
else: 
    print('not greater')
print('done')
```

    not greater
    done



```python
# Another example:

num = 53
print('before conditional...')
if num > 100:
    print(num, 'is greater than 100')
    
print('...after conditional')

# * use : after 'if' statment, always indent after 'if' statement
# * else statement is not necessary
```

    before conditional...
    ...after conditional



```python
# else/if example with 3 conditions:

num = -3

if num > 0:
    print(num, 'is positive')
elif num == 0:
    print(num, 'is zero')
else:
    print(num, 'is negative')
```

    -3 is negative



```python
if (1>0) and (-1 >= 0):
    print('both parts are true')
else:
    print('at least one part is false')
```

    at least one part is false



```python
if (1 > 0) and (-1 >= 0):
    print('at least one part is true')
else:
    print('at least one part is false')
```

    at least one part is false



```python
if (-1 > 0) and (-1 >= 0):
    print('at least one part is true')
else:
    print('both of these are false')
```

    both of these are false



```python
if (-1 > 0) or (-1 >= 0):
    print('at least one part is true')
else:
    print('both of these are false')
```

    both of these are false



```python

```


```python

```
```python
import numpy
```


```python
data = numpy.loadtxt(fname='Python_Data/inflammation-01.csv', delimiter =',')
max_inflammation_0 = numpy.amax(data, axis = 0)[0]

max_inflammation_20 = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Saspictious looking maxima!')
```

    Saspictious looking maxima!



```python
if max_inflammation_0 == 0 and max_inflammation_20 == 20:
    print('Saspictious looking maxima!')

elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
    print('Minima add up to zero!')
    
else:
    print('Seems OK!')

    # This looks to see if there is a healthy person in the stuyd (with no inflammation)
    # We see this isn't true.
```

    Saspictious looking maxima!



```python
data = numpy.loadtxt(fname = 'Python_Data/inflammation-03.csv', delimiter = ',')

max_inflammation_0 = numpy.amax(data, axis = 0)[0]

max_inflammation_20  = numpy.amax(data, axis = 0)[20]

if max_inflammation_0 == 0 and max_inflammation_20 ==20:
    print('Suspicious looking maxima!')
    
elif numpy.sum(numpy.amin(data, axis=0)) == 0:
    print('Minima add up to zero! -> Healthy Participant Alert!')
    
else:
    print('Seems OK!')
```

    Minima add up to zero! -> Healthy Participant Alert!



```python

```
## Functions

```python
# A function example
```


```python
# To change F to C values
fahrenheit_val = 99
celsius_val = ((fahrenheit_val -32)*(5/9))
print(celsius_val)
```

    37.22222222222222



```python
fahreneit_val2 = 43
celsius_val2 = ((fahreneit_val2-32)*(5/9))

print(celsius_val2)
```

    6.111111111111112



```python
def explicit_fahr_to_celsius(temp):
    #Assign the converted value to a variable
    converted = ((temp-32)*(5/9))
    #Return the values of the new variable
    return converted
```


```python
def fahr_to_celsius(temp):
    # Return converted values more efficiently using the return function without creating 
    # a new variable. This code does the same thing as the previous function but it is more 
    # explicit in explaining how the return command works. This combines the two previous lines of code.
    
    return((temp-32)*(5/9))
```


```python
fahr_to_celsius(32)
```




    0.0




```python
explicit_fahr_to_celsius(32)
```




    0.0




```python
# Both functions give the same result.
```


```python
print('Freezing point of water:', fahr_to_celsius(32), 'C')
print('Boiling point of water:', fahr_to_celsius(212), 'C')
```

    Freezing point of water: 0.0 C
    Boiling point of water: 100.0 C



```python
# Another function example:

def celsius_to_kelvin(temp_c):
    return temp_c + 273.15

print('Freezing point of water in Kelvin:', celsius_to_kelvin(0.))
```

    Freezing point of water in Kelvin: 273.15



```python
def fahr_to_kelvin(temp_f):
    temp_c = fahr_to_celsius(temp_f)
    temp_k = celsius_to_kelvin(temp_c)
    return temp_k

print('Boiling point of water in Kelvin:', fahr_to_kelvin(212.0))
```

    Boiling point of water in Kelvin: 373.15



```python
print('Again, temperature in Kelvin was:', temp_kelvin)
# Note that these variables are called local because they are in a function and do not exist outside of the function.
```

    Again, temperature in Kelvin was: 373.15



```python
# To make the variable permanent, not in function:

temp_kelvin = fahr_to_kelvin(212.0)
print('Temperature in Kelvin was:', temp_kelvin)
```

    Temperature in Kelvin was: 373.15



```python
temp_kelvin
```


```python
def print_temperatures():
    print('Temperature in Fahrenheit was:', temp_fahr)
    print('Temperature in Kelvin was:', temp_kelvin)
    
temp_fahr = 212.0
temp_kelvin = fahr_to_kelvin(temp_fahr)
    
print_temperatures()
```


```python
import numpy
import matplotlib.pyplot
import glob
```


```python
import numpy
import matplotlib.pyplot

%matplotlib inline  

def visualize(filename):
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    fig = matplotlib.pyplot.figure(figsize=(10.0, 3.0))
    
    axes1 = fig.add_subplot(1,3,1)
    axes2 = fig.add_subplot(1,3,2)
    axes3 = fig.add_subplot(1,3,3)
    
    axes1.set_ylabel('average')
    axes1.plot(numpy.mean(data, axis=0))
    
    axes2.set_ylabel('max')
    axes2.plot(numpy.max(data, axis=0))
    
    axes3.set_ylabel('min')
    axes3.plot(numpy.min(data, axis=0))
    
    fig.tight_layout()
    matplotlib.pyplot.show()

# Call your function
visualize('Python_Data/inflammation-01.csv')
```


![png](output_2_0.png)



```python
# a function:

def detect_problems(filename):
    
    data = numpy.loadtxt(fname = filename, delimiter = ',')
    
    if numpy.amax(data, axis = 0)[0] == 0 and numpy.amax(data, axis = 0)[20] == 20:
        print('Suspicious looking maxima!')
        # * we might just have line numbers and no actual data.
    
    elif numpy.sum(numpy.amin(data, axis = 0)) == 0:
        print('Minima add up to zero!')
    else:
        print('Seems OK!')

detect_problems('Python_Data/inflammation-01.csv')
```

    Suspicious looking maxima!



```python
import glob
import matplotlib.pyplot

filenames = sorted(glob.glob('Python_Data/inflammation*.csv'))

# For the first 3:
for filename in filenames[:3]:
    print(filename)
    visualize(filename)
    detect_problems(filename)

```

    Python_Data/inflammation-01.csv



![png](output_4_1.png)


    Suspicious looking maxima!
    Python_Data/inflammation-02.csv



![png](output_4_3.png)


    Suspicious looking maxima!
    Python_Data/inflammation-03.csv



![png](output_4_5.png)


    Minima add up to zero!



```python
# Let's create a new function

def offset_mean(data, target_mean_value):
    return(data - numpy.mean(data)) + target_mean_value
```


```python
# To test our function to see if it works, we can create testing data:

z = numpy.zeros((2,2))
print(offset_mean(z,3))

```

    [[3. 3.]
     [3. 3.]]



```python
data = numpy.loadtxt(fname = 'Python_Data/inflammation-01.csv', delimiter = ',')

print(offset_mean(data,0))
```

    [[-6.14875 -6.14875 -5.14875 ... -3.14875 -6.14875 -6.14875]
     [-6.14875 -5.14875 -4.14875 ... -5.14875 -6.14875 -5.14875]
     [-6.14875 -5.14875 -5.14875 ... -4.14875 -5.14875 -5.14875]
     ...
     [-6.14875 -5.14875 -5.14875 ... -5.14875 -5.14875 -5.14875]
     [-6.14875 -6.14875 -6.14875 ... -6.14875 -4.14875 -6.14875]
     [-6.14875 -6.14875 -5.14875 ... -5.14875 -5.14875 -6.14875]]



```python
print('original min, mean and max are :', 
      numpy.amin(data), 
      numpy.mean(data), 
      numpy.amax(data))

offset_data = offset_mean(data, 0)

print('min, mean, and max of offset data are:',
      numpy.amin(offset_data),
      numpy.mean(offset_data),
      numpy.amax(offset_data))
```

    original min, mean and max are : 0.0 6.14875 20.0
    min, mean, and max of offset data are: -6.14875 2.842170943040401e-16 13.85125



```python
print('std dev before and after:', numpy.std(data), numpy.std(offset_data))
```

    std dev before and after: 4.613833197118566 4.613833197118566



```python
print('difference in standard deviation before and after:',
     numpy.std(data) - numpy.std(offset_data))
```

    difference in standard deviation before and after: 0.0



```python
# offsest_mean(data, target_mean_value):
# return a new array containing the original data with its mean offset to match the desired value:

def offset_mean(data, target_mean_value):
    return(data - numpy.mean(data)) + target_mean_value
```


```python
def offset_mean(data, target_mean_value):
    """Return a new array containing the original data with its mean offset to match the desired value"""
    return(data - numpy.mean(data))  + target_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the original data with its mean offset to match the desired value
    



```python
def offset_mean(data, target_mean_value):
    """Return a new array containing the orginal data with its mean offset to match the desired value.
    
    Examples
    -----------
    
    >>> Offset_mean([1, 2, 3],0)
    array([-1., 0., 1.])
    """
    
    return(data - numpy.mean(data)) + target_mean_value
```


```python
help(offset_mean)
```

    Help on function offset_mean in module __main__:
    
    offset_mean(data, target_mean_value)
        Return a new array containing the orginal data with its mean offset to match the desired value.
        
        Examples
        -----------
        
        >>> Offset_mean([1, 2, 3],0)
        array([-1., 0., 1.])
    



```python
import numpy
numpy.loadtxt('Python_Data/inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python
def offset_mean(data, target_mean_value = 0.0):
    """Return a new array containing the orginal data with its mean offset to match the desired value, (0 by default).
    
    Examples
    -----------
    
    >>> Offset_mean([1, 2, 3])
    array([-1., 0., 1.])
    """
    
    return(data - numpy.mean(data)) + target_mean_value
```


```python
test_data = numpy.zeros((2,2))
print(offset_mean(test_data, 3))
```

    [[3. 3.]
     [3. 3.]]



```python
print(offset_mean(test_data))
```

    [[0. 0.]
     [0. 0.]]



```python
def display(a=1, b=2, c=3):
    print('a:', a, 'b:', b, 'c:', c)
    
print('no parametrs:')
display()
print('one parameter:')
display(55)
print('two parameters:')
display(55,66)
```

    no parametrs:
    a: 1 b: 2 c: 3
    one parameter:
    a: 55 b: 2 c: 3
    two parameters:
    a: 55 b: 66 c: 3



```python
print('only setting the value of c')
display(c=77)
```

    only setting the value of c
    a: 1 b: 2 c: 77



```python
help(numpy.loadtxt)
```

    Help on function loadtxt in module numpy:
    
    loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0, encoding='bytes', max_rows=None)
        Load data from a text file.
        
        Each row in the text file must have the same number of values.
        
        Parameters
        ----------
        fname : file, str, or pathlib.Path
            File, filename, or generator to read.  If the filename extension is
            ``.gz`` or ``.bz2``, the file is first decompressed. Note that
            generators should return byte strings for Python 3k.
        dtype : data-type, optional
            Data-type of the resulting array; default: float.  If this is a
            structured data-type, the resulting array will be 1-dimensional, and
            each row will be interpreted as an element of the array.  In this
            case, the number of columns used must match the number of fields in
            the data-type.
        comments : str or sequence of str, optional
            The characters or list of characters used to indicate the start of a
            comment. None implies no comments. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is '#'.
        delimiter : str, optional
            The string used to separate values. For backwards compatibility, byte
            strings will be decoded as 'latin1'. The default is whitespace.
        converters : dict, optional
            A dictionary mapping column number to a function that will parse the
            column string into the desired value.  E.g., if column 0 is a date
            string: ``converters = {0: datestr2num}``.  Converters can also be
            used to provide a default value for missing data (but see also
            `genfromtxt`): ``converters = {3: lambda s: float(s.strip() or 0)}``.
            Default: None.
        skiprows : int, optional
            Skip the first `skiprows` lines, including comments; default: 0.
        usecols : int or sequence, optional
            Which columns to read, with 0 being the first. For example,
            ``usecols = (1,4,5)`` will extract the 2nd, 5th and 6th columns.
            The default, None, results in all columns being read.
        
            .. versionchanged:: 1.11.0
                When a single column has to be read it is possible to use
                an integer instead of a tuple. E.g ``usecols = 3`` reads the
                fourth column the same way as ``usecols = (3,)`` would.
        unpack : bool, optional
            If True, the returned array is transposed, so that arguments may be
            unpacked using ``x, y, z = loadtxt(...)``.  When used with a structured
            data-type, arrays are returned for each field.  Default is False.
        ndmin : int, optional
            The returned array will have at least `ndmin` dimensions.
            Otherwise mono-dimensional axes will be squeezed.
            Legal values: 0 (default), 1 or 2.
        
            .. versionadded:: 1.6.0
        encoding : str, optional
            Encoding used to decode the inputfile. Does not apply to input streams.
            The special value 'bytes' enables backward compatibility workarounds
            that ensures you receive byte arrays as results if possible and passes
            'latin1' encoded strings to converters. Override this value to receive
            unicode arrays and pass strings as input to converters.  If set to None
            the system default is used. The default value is 'bytes'.
        
            .. versionadded:: 1.14.0
        max_rows : int, optional
            Read `max_rows` lines of content after `skiprows` lines. The default
            is to read all the lines.
        
            .. versionadded:: 1.16.0
        
        Returns
        -------
        out : ndarray
            Data read from the text file.
        
        See Also
        --------
        load, fromstring, fromregex
        genfromtxt : Load data with missing values handled as specified.
        scipy.io.loadmat : reads MATLAB data files
        
        Notes
        -----
        This function aims to be a fast reader for simply formatted files.  The
        `genfromtxt` function provides more sophisticated handling of, e.g.,
        lines with missing values.
        
        .. versionadded:: 1.10.0
        
        The strings produced by the Python float.hex method can be used as
        input for floats.
        
        Examples
        --------
        >>> from io import StringIO   # StringIO behaves like a file object
        >>> c = StringIO(u"0 1\n2 3")
        >>> np.loadtxt(c)
        array([[0., 1.],
               [2., 3.]])
        
        >>> d = StringIO(u"M 21 72\nF 35 58")
        >>> np.loadtxt(d, dtype={'names': ('gender', 'age', 'weight'),
        ...                      'formats': ('S1', 'i4', 'f4')})
        array([(b'M', 21, 72.), (b'F', 35, 58.)],
              dtype=[('gender', 'S1'), ('age', '<i4'), ('weight', '<f4')])
        
        >>> c = StringIO(u"1,0,2\n3,0,4")
        >>> x, y = np.loadtxt(c, delimiter=',', usecols=(0, 2), unpack=True)
        >>> x
        array([1., 3.])
        >>> y
        array([2., 4.])
    



```python
numpy.loadtxt('Python_Data/inflammation-01.csv', delimiter = ',')
```




    array([[0., 0., 1., ..., 3., 0., 0.],
           [0., 1., 2., ..., 1., 0., 1.],
           [0., 1., 1., ..., 2., 1., 1.],
           ...,
           [0., 1., 1., ..., 1., 1., 1.],
           [0., 0., 0., ..., 0., 2., 0.],
           [0., 0., 1., ..., 1., 1., 0.]])




```python

```

## Defensive Programming

```python
# To make sure our functions do what we want them to do. 
# Just because there is no error, does not mean the function accomplishes the goal.
# To do this, we add asssertions to our code. 
# Assertion: a statement that something must be true at a certain point in the gporam
# When python sees one, python does the function. If false, it stops.
```


```python
numbers = [0.5, 2.3, 0.7, -0.001, 4.4]
total = 0.0
for num in numbers:
    assert num > 0.0, 'Data should only contain positive values'
    total += num
print('total is:', total)
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-3-d596a7ea0a7e> in <module>
          2 total = 0.0
          3 for num in numbers:
    ----> 4     assert num > 0.0, 'Data should only contain positive values'
          5     total += num
          6 print('total is:', total)


    AssertionError: Data should only contain positive values



```python
# Because we told python that data should only contain positve values (assert) we get an assertion error.
# A lot of code exists to check to make sure the other code is running correctly.
```


```python
# 3 types of assertions:
# pre-condition: someting that must be true at the start of a function.
# post-condition: something is guaranteed to be true at the end of a function.
# invariant: something must be true at all times during the execution of a function, within a code region.

```


```python
# Example:
# Assume:

# (x0, y0, x1, y1)

def normalize_rectangle(rect):
    '''Normalizes a rectangle so that it is at the origin and 1.0 units long on its longest axis.
    input should be of the format(x0, y0, x1, y1).
    (x0, y0) and (x1, y1) define the lower left and uppre right corners of the rectangle respectively.'''
    assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
    x0, y0, x1, y1 = rect
    assert x0 < x1, 'Invalid X coordinates'
    assert y0 < y1, 'Invalid Y coordinates'
    
    dx = x1-x0
    dy = y1-y0
    if dy > dx:
        scaled = dy/dx
        upper_x, upper_y = 1.0, scaled
    else:
        scaled = dx/dy
        upper_x, upper_y = scaled, 1.0
        
        assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
        assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'
        
        return(0, 0, upper_x, upper_y)
```


```python
print(normalize_rectangle((0.0,1.0, 2.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-31-e122f79e7e85> in <module>
    ----> 1 print(normalize_rectangle((0.0,1.0, 2.0)))
    

    <ipython-input-24-5236a5df1673> in normalize_rectangle(rect)
          8     input should be of the format(x0, y0, x1, y1).
          9     (x0, y0) and (x1, y1) define the lower left and uppre right corners of the rectangle respectively.'''
    ---> 10     assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
         11     x0, y0, x1, y1 = rect
         12     assert x0 < x1, 'Invalid X coordinates'


    AssertionError: Rectangles must contain 4 coordinates



```python
# Assertion error: only contains 3 coordinates
```


```python
print(normalize_rectangle((4.0, 2.0, 1.0, 5.0 )))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-32-ff4f62975a5e> in <module>
    ----> 1 print(normalize_rectangle((4.0, 2.0, 1.0, 5.0 )))
    

    <ipython-input-24-5236a5df1673> in normalize_rectangle(rect)
         10     assert len(rect) == 4, 'Rectangles must contain 4 coordinates'
         11     x0, y0, x1, y1 = rect
    ---> 12     assert x0 < x1, 'Invalid X coordinates'
         13     assert y0 < y1, 'Invalid Y coordinates'
         14 


    AssertionError: Invalid X coordinates



```python
# assertion error: x0 is not less than x1
```


```python
print(normalize_rectangle((0.0, 0.0, 1.0, 5.0)))
```

    None



```python
print(normalize_rectangle((0.0, 0.0, 5.0, 1.0)))
```


    ---------------------------------------------------------------------------

    AssertionError                            Traceback (most recent call last)

    <ipython-input-34-1337bef8f4bf> in <module>
    ----> 1 print(normalize_rectangle((0.0, 0.0, 5.0, 1.0)))
    

    <ipython-input-24-5236a5df1673> in normalize_rectangle(rect)
         22         upper_x, upper_y = scaled, 1.0
         23 
    ---> 24         assert 0 < upper_x <= 1.0, 'Calculated upper x coordinate invalid'
         25         assert 0 < upper_y <= 1.0, 'Calculated upper y coordinate invalid'
         26 


    AssertionError: Calculated upper x coordinate invalid

## Transcribing DNA into RNA

```python
# Prompt the user to enter the input fasta file name

input_file_name = input("Enter the name of the input fasta file")
```

    Enter the name of the input fasta file sequence (2).txt



```python
# Open the input fasta file and read the DNA sequence

with open(input_file_name, "r") as input_file:
    dna_sequence = ""
    for line in input_file:
        if line.startswith(">"):
            continue
        dna_sequence += line.strip()
```


```python
# Transcribe the DNA to RNA

rna_sequence = ""
for nucleotide in dna_sequence:
    if nucleotide == "T":
        rna_sequence += "U"
    else:
        rna_sequence += nucleotide
```


```python
# Prompt user to enter the output file name

output_file_name = input("Enter the name of the output file: ")
```

    Enter the name of the output file:  Fibrillin_RNA.txt



```python
# Save the RNA sequence to a text file

with open(output_file_name, "w") as output_file:
    output_file.write(rna_sequence)
    print("The RNA sequence has been saved to {output_file_name}")
```

    The RNA sequence has been saved to {output_file_name}



```python
print(f"RNA: {rna_sequence}")
```

    RNA: AUGCGUCGAGGGCGUCUGCUGGAGAUCGCCCUGGGAUUUACCGUGCUUUUAGCGUCCUACACGAGCCAUGGGGCGGACGCCAAUUUGGAGGCUGGGAACGUGAAGGAAACCAGAGCCAGUCGGGCCAAGAGAAGAGGCGGUGGAGGACACGACGCGCUUAAAGGACCCAAUGUCUGUGGAUCACGUUAUAAUGCUUACUGUUGCCCUGGAUGGAAAACCUUACCUGGCGGAAAUCAGUGUAUUGUCCCCAUUUGCCGGCAUUCCUGUGGGGAUGGAUUUUGUUCGAGGCCAAAUAUGUGCACUUGCCCAUCUGGUCAGAUAGCUCCUUCCUGUGGCUCCAGAUCCAUACAACACUGCAAUAUUCGCUGUAUGAAUGGAGGUAGCUGCAGUGACGAUCACUGUCUAUGCCAGAAAGGAUACAUAGGGACUCACUGUGGACAACCUGUUUGUGAAAGUGGCUGUCUCAAUGGAGGAAGGUGUGUGGCCCCAAAUCGAUGUGCAUGCACUUACGGAUUUACUGGACCCCAGUGUGAAAGAGAUUACAGGACAGGCCCAUGUUUUACUGUGAUCAGCAACCAGAUGUGCCAGGGACAACUCAGCGGGAUUGUCUGCACAAAAACGCUCUGCUGUGCCACAGUCGGCCGAGCCUGGGGCCACCCCUGUGAGAUGUGUCCUGCCCAGCCUCACCCCUGCCGCCGUGGCUUCAUUCCAAAUAUCCGCACGGGAGCUUGUCAAGAUGUGGAUGAAUGCCAGGCCAUCCCCGGGCUCUGUCAGGGAGGAAAUUGCAUUAAUACUGUUGGGUCUUUUGAGUGCAAAUGCCCUGCUGGACACAAACUUAAUGAAGUGUCACAAAAAUGUGAAGAUAUUGAUGAAUGCAGCACCAUUCCUGGAAUCUGUGAAGGGGGUGAAUGUACAAACACAGUCAGCAGUUACUUUUGCAAAUGUCCCCCUGGUUUUUACACCUCUCCAGAUGGUACCAGAUGCAUAGAUGUUCGCCCAGGAUACUGUUACACAGCUCUGACAAACGGGCGCUGCUCUAACCAGCUGCCACAGUCCAUAACCAAAAUGCAGUGCUGCUGUGAUGCCGGCCGAUGCUGGUCUCCAGGGGUCACUGUCGCCCCUGAGAUGUGUCCCAUCAGAGCAACCGAGGAUUUCAACAAGCUGUGCUCUGUUCCUAUGGUAAUUCCUGGGAGACCAGAAUAUCCUCCCCCACCCCUUGGCCCCAUUCCUCCAGUUCUCCCUGUUCCUCCUGGCUUUCCUCCUGGACCUCAAAUUCCGGUCCCUCGACCACCAGUGGAAUAUCUGUAUCCAUCUCGGGAGCCACCAAGGGUGCUGCCAGUAAACGUUACUGAUUACUGCCAGUUGGUCCGCUAUCUCUGUCAAAAUGGACGCUGCAUUCCAACUCCUGGGAGUUACCGGUGUGAGUGCAACAAAGGGUUCCAGCUGGACCUCCGUGGGGAGUGUAUUGAUGUUGAUGAAUGUGAGAAAAACCCCUGUGCUGGUGGUGAGUGUAUUAACAACCAGGGUUCGUACACCUGUCAGUGCCGAGCUGGAUAUCAGAGCACACUCACGCGGACAGAAUGCCGAGACAUUGAUGAGUGUUUACAGAAUGGCCGGAUCUGCAAUAAUGGACGCUGCAUCAACACAGAUGGCAGUUUUCAUUGCGUGUGUAAUGCGGGCUUUCAUGUUACACGAGAUGGGAAGAACUGUGAAGAUAUGGAUGAAUGCAGCAUAAGGAACAUGUGCCUUAAUGGAAUGUGUAUCAAUGAAGAUGGCAGUUUUAAAUGUAUUUGCAAACCUGGAUUCCAGCUGGCAUCAGAUGGACGUUAUUGCAAAGACAUUAACGAGUGUGAAACCCCUGGGAUCUGCAUGAAUGGGCGUUGCGUCAACACUGAUGGCUCCUACAGAUGUGAAUGCUUCCCUGGACUGGCUGUGGGUCUGGAUGGCCGUGUGUGUGUUGACACACACAUGCGGAGCACAUGCUAUGGUGGAUACAAGAGAGGCCAGUGUAUCAAACCUUUGUUUGGUGCUGUCACUAAAUCUGAAUGCUGUUGCGCCAGCACUGAGUAUGCAUUUGGGGAACCUUGCCAGCCGUGUCCUGCACAGAAUUCAGCGGAAUAUCAGGCACUCUGCAGCAGUGGGCCAGGAAUGACGUCAGCAGGCAGUGAUAUAAAUGAAUGUGCACUAGAUCCUGAUAUUUGCCCAAAUGGAAUCUGUGAAAACCUUCGUGGGACCUAUAAAUGUAUAUGCAAUUCAGGAUAUGAAGUGGAUUCAACUGGGAAAAACUGCGUUGAUAUUAAUGAAUGUGUACUGAACAGUCUCCUUUGUGACAAUGGACAAUGUAGAAAUACUCCUGGAAGUUUUGUCUGUACCUGCCCCAAGGGAUUUAUCUACAAACCUGAUCUAAAAACAUGUGAAGACAUUGAUGAAUGCGAAUCAAGUCCUUGCAUUAAUGGAGUCUGCAAGAACAGCCCAGGCUCUUUUAUUUGUGAAUGUUCUUCUGAAAGUACUUUGGAUCCAACAAAAACCAUCUGCAUAGAAACCAUCAAGGGCACUUGCUGGCAGACUGUCAUUGAUGGGCGAUGUGAGAUCAACAUCAAUGGAGCCACCUUAAAGUCCCAGUGCUGCUCCUCCCUCGGUGCUGCGUGGGGAAGCCCGUGCACCCUAUGCCAAGUUGAUCCCAUAUGUGGUAAAGGGUACUCAAGAAUUAAAGGAACACAAUGUGAAGAUAUAGAUGAAUGUGAAGUGUUCCCAGGAGUGUGUAAAAAUGGCCUGUGUGUUAACACUAGGGGGUCAUUCAAGUGUCAGUGUCCCAGUGGAAUGACUUUGGAUGCCACAGGAAGGAUCUGUCUUGAUAUCCGCCUGGAAACCUGCUUCCUGAGGUACGAGGACGAGGAGUGCACCCUGCCUAUUGCUGGCCGCCACCGCAUGGACGCCUGCUGCUGCUCCGUCGGGGCAGCCUGGGGUACUGAGGAAUGCGAGGAGUGUCCCAUGAGAAAUACUCCUGAGUACGAGGAGCUGUGUCCGAGAGGACCCGGAUUUGCCACAAAAGAAAUUACAAAUGGAAAGCCUUUCUUCAAAGAUAUCAAUGAGUGCAAGAUGAUACCCAGCCUCUGCACCCACGGCAAGUGCAGAAACACCAUUGGCAGCUUUAAGUGCAGGUGUGACAGCGGCUUUGCUCUUGAUUCUGAAGAAAGGAACUGCACAGACAUUGACGAAUGCCGCAUAUCUCCUGACCUCUGUGGCAGAGGCCAGUGUGUGAACACCCCUGGGGACUUUGAAUGCAAGUGUGACGAAGGCUAUGAAAGUGGAUUCAUGAUGAUGAAGAACUGCAUGGAUAUUGAUGAGUGUCAGAGAGAUCCUCUCCUAUGCCGAGGUGGUGUUUGCCAUAACACAGAGGGAAGUUACCGCUGUGAAUGCCCGCCUGGCCAUCAGCUGUCCCCCAACAUCUCCGCGUGUAUCGACAUCAAUGAAUGUGAGCUGAGUGCACACCUGUGCCCCAAUGGCCGUUGCGUGAACCUCAUAGGGAAGUAUCAGUGUGCCUGCAACCCUGGCUACCAUUCAACUCCCGAUAGGCUAUUUUGUGUUGACAUUGAUGAAUGCAGCAUAAUGAAUGGUGGUUGUGAAACCUUCUGCACAAACUCUGAAGGCAGCUAUGAAUGUAGCUGUCAGCCGGGAUUUGCACUAAUGCCUGACCAGAGAUCAUGCACCGACAUCGAUGAGUGUGAAGAUAAUCCCAAUAUCUGUGAUGGUGGUCAGUGCACAAAUAUCCCUGGAGAGUACAGGUGCUUGUGUUAUGAUGGAUUCAUGGCAUCUGAAGACAUGAAGACUUGUGUAGAUGUCAAUGAGUGUGACCUGAAUCCAAAUAUCUGCCUAAGUGGGACCUGUGAAAACACGAAAGGCUCAUUUAUCUGCCACUGUGAUAUGGGCUACUCCGGCAAAAAAGGAAAAACUGGCUGUACAGACAUCAAUGAAUGUGAAAUUGGAGCACACAACUGUGGCAAACAUGCUGUAUGUACCAAUACAGCAGGAAGCUUCAAAUGUAGCUGCAGUCCCGGGUGGAUUGGAGAUGGCAUUAAGUGCACUGAUCUGGACGAAUGUUCCAAUGGAACCCAUAUGUGCAGCCAGCAUGCAGACUGCAAGAAUACCAUGGGAUCUUACCGCUGUCUGUGCAAGGAAGGAUACACAGGUGAUGGCUUCACUUGUACAGACCUUGAUGAGUGCUCUGAGAACCUGAAUCUCUGUGGCAAUGGCCAGUGCCUCAAUGCACCAGGAGGAUACCGCUGUGAAUGCGACAUGGGCUUCGUGCCCAGUGCUGACGGGAAAGCCUGUGAAGAUAUUGAUGAGUGCUCCCUUCCGAACAUCUGUGUCUUUGGAACUUGCCACAACCUCCCUGGCCUGUUCCGCUGUGAGUGUGAGAUAGGCUACGAACUGGACAGAAGCGGCGGGAACUGCACAGAUGUGAAUGAAUGCCUGGAUCCAACCACGUGCAUCAGUGGGAACUGUGUCAACACUCCAGGCAGCUAUAUCUGUGACUGCCCACCUGAUUUUGAACUGAACCCAACUCGAGUUGGCUGUGUUGAUACCCGCUCUGGAAAUUGCUAUUUGGAUAUUCGACCUCGAGGAGACAAUGGAGAUACAGCCUGCAGCAAUGAAAUUGGAGUUGGUGUUUCCAAAGCUUCCUGCUGCUGUUCUCUGGGUAAAGCCUGGGGUACUCCUUGUGAGAUGUGUCCUGCUGUGAACACAUCCGAGUACAAAAUUCUUUGUCCUGGAGGGGAAGGUUUCCGACCAAAUCCUAUCACCGUUAUAUUGGAAGAUAUUGAUGAGUGCCAGGAGCUACCAGGGCUGUGCCAAGGAGGAAAAUGUAUCAACACCUUUGGGAGUUUCCAGUGCCGCUGUCCAACCGGCUACUACCUGAAUGAAGAUACACGAGUGUGUGAUGAUGUGAAUGAAUGUGAGACUCCUGGAAUCUGUGGUCCAGGGACAUGUUACAACACCGUUGGCAACUACACCUGUAUCUGUCCUCCAGACUACAUGCAAGUGAAUGGGGGAAAUAAUUGCAUGGAUAUGAGAAGAAGUUUGUGCUACAGAAACUACUAUGCUGACAACCAGACCUGUGAUGGAGAAUUGUUAUUCAACAUGACCAAGAAGAUGUGCUGCUGUUCCUACAACAUUGGCCGGGCGUGGAACAAGCCCUGUGAACAGUGUCCCAUCCCAAGUACAGAUGAGUUUGCUACACUCUGUGGAAGUCAAAGGCCAGGCUUUGUCAUCGACAUUUAUACCGGUUUACCCGUUGAUAUUGAUGAGUGCCGGGAGAUCCCAGGGGUCUGUGAAAAUGGAGUGUGUAUCAACAUGGUUGGCAGCUUCCGAUGUGAAUGUCCAGUGGGAUUCUUCUAUAAUGACAAGUUGUUGGUUUGUGAAGAUAUUGACGAGUGUCAGAACGGCCCAGUGUGCCAGCGCAACGCCGAAUGCAUCAACACUGCAGGCAGCUACCGCUGUGACUGUAAGCCCGGCUACCGCUUCACCUCCACAGGACAGUGCAAUGAUCGUAAUGAAUGUCAAGAAAUCCCCAAUAUAUGCAGUCAUGGGCAGUGCAUUGACACAGUUGGAAGCUUUUAUUGCCUUUGCCACACUGGUUUUAAAACAAAUGAUGACCAAACCAUGUGCUUGGACAUAAAUGAAUGUGAAAGAGAUGCCUGUGGGAAUGGAACUUGCCGGAACACAAUUGGUUCCUUCAACUGCCGCUGCAAUCAUGGUUUCAUCCUUUCUCACAACAAUGACUGUAUAGAUGUUGAUGAAUGUGCAAGUGGAAAUGGGAAUCUUUGCAGAAAUGGCCAAUGCAUUAAUACAGUGGGGUCUUUCCAGUGCCAGUGCAAUGAAGGCUAUGAGGUGGCUCCAGAUGGGAGGACCUGUGUGGAUAUCAAUGAAUGUCUUCUAGAACCCAGAAAAUGUGCACCAGGUACCUGUCAAAACUUGGAUGGGUCCUACAGAUGCAUUUGCCCACCUGGAUACAGUCUUCAAAAUGAGAAGUGUGAAGAUAUUGAUGAGUGUGUCGAAGAGCCAGAAAUUUGUGCCCUGGGCACAUGCAGUAACACUGAAGGCAGCUUCAAAUGUCUGUGUCCAGAAGGGUUUUCCUUGUCCUCCAGUGGAAGAAGGUGCCAAGAUUUGCGAAUGAGCUACUGUUAUGCGAAGUUUGAAGGAGGAAAGUGUUCAUCACCCAAAUCCAGAAAUCACUCCAAGCAGGAAUGCUGCUGUGCCUUGAAGGGAGAAGGCUGGGGAGACCCCUGCGAGCUCUGCCCCACGGAACCUGAUGAGGCCUUCCGCCAGAUAUGUCCUUAUGGAAGUGGGAUCAUCGUGGGACCUGAUGAUUCAGCAGUUGAUAUGGACGAAUGCAAAGAACCCGAUGUCUGUAAACAUGGACAGUGCAUCAAUACAGAUGGUUCCUAUCGCUGCGAGUGUCCCUUUGGUUAUAUUCUAGCAGGGAAUGAAUGUGUAGAUACUGAUGAAUGUUCUGUUGGCAAUCCUUGUGGAAAUGGAACCUGCAAGAAUGUGAUUGGAGGUUUUGAAUGCACCUGCGAGGAGGGAUUUGAGCCCGGUCCAAUGAUGACAUGUGAAGAUAUAAAUGAAUGUGCCCAGAAUCCUCUGCUCUGUGCCUUCCGAUGUGUGAACACUUAUGGGUCAUAUGAAUGCAAAUGUCCCGUGGGAUAUGUGCUCAGAGAAGACCGUAGGAUGUGCAAAGAUGAGGAUGAGUGUGAAGAGGGAAAACAUGACUGUACUGAAAAACAAAUGGAAUGCAAGAACCUCAUUGGCACAUAUAUGUGCAUCUGUGGACCCGGGUAUCAGCGGAGACCUGAUGGAGAAGGCUGUGUAGAUGAGAAUGAAUGUCAGACGAAGCCAGGGAUCUGUGAGAAUGGGCGCUGCCUCAACACCCGUGGGAGCUACACCUGUGAGUGUAAUGAUGGGUUUACCGCCAGCCCCAACCAGGACGAGUGCCUUGACAAUCGGGAAGGGUACUGCUUCACAGAGGUGCUACAAAACAUGUGUCAGAUCGGCUCCAGCAACAGGAACCCCGUCACCAAAUCGGAAUGCUGCUGUGACGGAGGGAGAGGCUGGGGUCCCCACUGUGAGAUCUGCCCUUUCCAGGGGACUGUGGCUUUCAAGAAACUCUGUCCCCAUGGCCGAGGAUUCAUGACCAAUGGAGCAGAUAUCGAUGAAUGCAAGGUUAUUCACGAUGUUUGCCGAAAUGGGGAAUGUGUCAAUGACAGAGGAUCAUAUCAUUGCAUUUGUAAAACUGGGUACACUCCAGAUAUAACUGGGACUUCCUGUGUAGAUCUGAACGAGUGCAACCAGGCUCCCAAACCCUGCAAUUUUAUCUGCAAAAACACAGAAGGGAGUUACCAGUGUUCAUGCCCGAAAGGCUACAUUCUGCAAGAGGAUGGAAGGAGCUGCAAAGAUCUUGAUGAGUGUGCAACCAAGCAACACAACUGCCAGUUCCUAUGUGUUAACACCAUUGGCGGCUUCACAUGCAAAUGUCCUCCCGGAUUUACCCAACACCAUACGUCCUGCAUUGAUAACAAUGAAUGCACCUCUGACAUCAAUCUGUGCGGGUCUAAGGGCAUUUGCCAGAACACUCCUGGAAGCUUCACCUGUGAAUGCCAGCGGGGAUUCUCACUUGAUCAGACCGGCUCCAGCUGUGAAGACGUGGACGAGUGUGAGGGUAACCACCGCUGCCAGCAUGGCUGCCAGAACAUCAUUGGGGGCUACAGGUGCAGCUGCCCCCAGGGCUACCUCCAGCACUACCAGUGGAACCAGUGUGUUGAUGAAAACGAAUGCCUCAGCGCUCACAUCUGCGGAGGAGCCUCCUGUCACAACACCCUGGGGAGCUACAAGUGCAUGUGUCCCGCCGGCUUCCAGUAUGAACAGUUCAGUGGAGGAUGCCAAGACAUCAAUGAAUGUGGCUCUGCGCAGGCCCCCUGCAGCUAUGGCUGUUCCAAUACCGAGGGCGGUUACCUGUGUGGCUGUCCACCUGGUUACUUCCGCAUAGGCCAAGGGCACUGUGUUUCUGGAAUGGGCAUGGGCCGAGGAAACCCAGAGCCACCUGUCAGUGGUGAAAUGGAUGACAAUUCACUCUCCCCAGAGGCUUGUUACGAGUGUAAGAUCAAUGGCUACCCCAAACGGGGCAGGAAACGGAGAAGCACAAACGAAACUGAUGCCUCCAAUAUCGAGGAUCAGUCUGAGACAGAAGCCAAUGUGAGUCUUGCAAGUUGGGAUGUUGAGAAGACAGCCAUCUUUGCUUUCAAUAUUUCCCACGUCAGUAACAAGGUUCGAAUCCUAGAACUCCUUCCAGCUCUUACAACUCUGACGAAUCACAACAGAUACUUGAUCGAAUCUGGAAAUGAAGAUGGCUUCUUUAAAAUCAACCAAAAGGAAGGGAUCAGCUACCUCCACUUCACAAAGAAGAAGCCAGUGGCUGGAACCUAUUCAUUACAAAUCAGUAGUACUCCACUUUAUAAAAAGAAAGAACUUAACCAACUAGAAGACAAAUAUGACAAAGACUACCUCAGUGGUGAACUGGGUGAUAAUCUGAAGAUGAAAAUCCAGGUUUUGCUUCAUUAAAUGCGUCGAGGGCGUCUGCUGGAGAUCGCCCUGGGAUUUACCGUGCUUUUAGCGUCCUACACGAGCCAUGGGGCGGACGCCAAUUUGGAGGCUGGGAACGUGAAGGAAACCAGAGCCAGUCGGGCCAAGAGAAGAGGCGGUGGAGGACACGACGCGCUUAAAGGACCCAAUGUCUGUGGAUCACGUUAUAAUGCUUACUGUUGCCCUGGAUGGAAAACCUUACCUGGCGGAAAUCAGUGUAUUGUCCCCAUUUGCCGGCAUUCCUGUGGGGAUGGAUUUUGUUCGAGGCCAAAUAUGUGCACUUGCCCAUCUGGUCAGAUAGCUCCUUCCUGUGGCUCCAGAUCCAUACAACACUGCAAUAUUCGCUGUAUGAAUGGAGGUAGCUGCAGUGACGAUCACUGUCUAUGCCAGAAAGGAUACAUAGGGACUCACUGUGGACAACCUGUUUGUGAAAGUGGCUGUCUCAAUGGAGGAAGGUGUGUGGCCCCAAAUCGAUGUGCAUGCACUUACGGAUUUACUGGACCCCAGUGUGAAAGAGAUUACAGGACAGGCCCAUGUUUUACUGUGAUCAGCAACCAGAUGUGCCAGGGACAACUCAGCGGGAUUGUCUGCACAAAAACGCUCUGCUGUGCCACAGUCGGCCGAGCCUGGGGCCACCCCUGUGAGAUGUGUCCUGCCCAGCCUCACCCCUGCCGCCGUGGCUUCAUUCCAAAUAUCCGCACGGGAGCUUGUCAAGAUGUGGAUGAAUGCCAGGCCAUCCCCGGGCUCUGUCAGGGAGGAAAUUGCAUUAAUACUGUUGGGUCUUUUGAGUGCAAAUGCCCUGCUGGACACAAACUUAAUGAAGUGUCACAAAAAUGUGAAGAUAUUGAUGAAUGCAGCACCAUUCCUGGAAUCUGUGAAGGGGGUGAAUGUACAAACACAGUCAGCAGUUACUUUUGCAAAUGUCCCCCUGGUUUUUACACCUCUCCAGAUGGUACCAGAUGCAUAGAUGUUCGCCCAGGAUACUGUUACACAGCUCUGACAAACGGGCGCUGCUCUAACCAGCUGCCACAGUCCAUAACCAAAAUGCAGUGCUGCUGUGAUGCCGGCCGAUGCUGGUCUCCAGGGGUCACUGUCGCCCCUGAGAUGUGUCCCAUCAGAGCAACCGAGGAUUUCAACAAGCUGUGCUCUGUUCCUAUGGUAAUUCCUGGGAGACCAGAAUAUCCUCCCCCACCCCUUGGCCCCAUUCCUCCAGUUCUCCCUGUUCCUCCUGGCUUUCCUCCUGGACCUCAAAUUCCGGUCCCUCGACCACCAGUGGAAUAUCUGUAUCCAUCUCGGGAGCCACCAAGGGUGCUGCCAGUAAACGUUACUGAUUACUGCCAGUUGGUCCGCUAUCUCUGUCAAAAUGGACGCUGCAUUCCAACUCCUGGGAGUUACCGGUGUGAGUGCAACAAAGGGUUCCAGCUGGACCUCCGUGGGGAGUGUAUUGAUGUUGAUGAAUGUGAGAAAAACCCCUGUGCUGGUGGUGAGUGUAUUAACAACCAGGGUUCGUACACCUGUCAGUGCCGAGCUGGAUAUCAGAGCACACUCACGCGGACAGAAUGCCGAGACAUUGAUGAGUGUUUACAGAAUGGCCGGAUCUGCAAUAAUGGACGCUGCAUCAACACAGAUGGCAGUUUUCAUUGCGUGUGUAAUGCGGGCUUUCAUGUUACACGAGAUGGGAAGAACUGUGAAGAUAUGGAUGAAUGCAGCAUAAGGAACAUGUGCCUUAAUGGAAUGUGUAUCAAUGAAGAUGGCAGUUUUAAAUGUAUUUGCAAACCUGGAUUCCAGCUGGCAUCAGAUGGACGUUAUUGCAAAGACAUUAACGAGUGUGAAACCCCUGGGAUCUGCAUGAAUGGGCGUUGCGUCAACACUGAUGGCUCCUACAGAUGUGAAUGCUUCCCUGGACUGGCUGUGGGUCUGGAUGGCCGUGUGUGUGUUGACACACACAUGCGGAGCACAUGCUAUGGUGGAUACAAGAGAGGCCAGUGUAUCAAACCUUUGUUUGGUGCUGUCACUAAAUCUGAAUGCUGUUGCGCCAGCACUGAGUAUGCAUUUGGGGAACCUUGCCAGCCGUGUCCUGCACAGAAUUCAGCGGAAUAUCAGGCACUCUGCAGCAGUGGGCCAGGAAUGACGUCAGCAGGCAGUGAUAUAAAUGAAUGUGCACUAGAUCCUGAUAUUUGCCCAAAUGGAAUCUGUGAAAACCUUCGUGGGACCUAUAAAUGUAUAUGCAAUUCAGGAUAUGAAGUGGAUUCAACUGGGAAAAACUGCGUUGAUAUUAAUGAAUGUGUACUGAACAGUCUCCUUUGUGACAAUGGACAAUGUAGAAAUACUCCUGGAAGUUUUGUCUGUACCUGCCCCAAGGGAUUUAUCUACAAACCUGAUCUAAAAACAUGUGAAGACAUUGAUGAAUGCGAAUCAAGUCCUUGCAUUAAUGGAGUCUGCAAGAACAGCCCAGGCUCUUUUAUUUGUGAAUGUUCUUCUGAAAGUACUUUGGAUCCAACAAAAACCAUCUGCAUAGAAACCAUCAAGGGCACUUGCUGGCAGACUGUCAUUGAUGGGCGAUGUGAGAUCAACAUCAAUGGAGCCACCUUAAAGUCCCAGUGCUGCUCCUCCCUCGGUGCUGCGUGGGGAAGCCCGUGCACCCUAUGCCAAGUUGAUCCCAUAUGUGGUAAAGGGUACUCAAGAAUUAAAGGAACACAAUGUGAAGAUAUAGAUGAAUGUGAAGUGUUCCCAGGAGUGUGUAAAAAUGGCCUGUGUGUUAACACUAGGGGGUCAUUCAAGUGUCAGUGUCCCAGUGGAAUGACUUUGGAUGCCACAGGAAGGAUCUGUCUUGAUAUCCGCCUGGAAACCUGCUUCCUGAGGUACGAGGACGAGGAGUGCACCCUGCCUAUUGCUGGCCGCCACCGCAUGGACGCCUGCUGCUGCUCCGUCGGGGCAGCCUGGGGUACUGAGGAAUGCGAGGAGUGUCCCAUGAGAAAUACUCCUGAGUACGAGGAGCUGUGUCCGAGAGGACCCGGAUUUGCCACAAAAGAAAUUACAAAUGGAAAGCCUUUCUUCAAAGAUAUCAAUGAGUGCAAGAUGAUACCCAGCCUCUGCACCCACGGCAAGUGCAGAAACACCAUUGGCAGCUUUAAGUGCAGGUGUGACAGCGGCUUUGCUCUUGAUUCUGAAGAAAGGAACUGCACAGACAUUGACGAAUGCCGCAUAUCUCCUGACCUCUGUGGCAGAGGCCAGUGUGUGAACACCCCUGGGGACUUUGAAUGCAAGUGUGACGAAGGCUAUGAAAGUGGAUUCAUGAUGAUGAAGAACUGCAUGGAUAUUGAUGAGUGUCAGAGAGAUCCUCUCCUAUGCCGAGGUGGUGUUUGCCAUAACACAGAGGGAAGUUACCGCUGUGAAUGCCCGCCUGGCCAUCAGCUGUCCCCCAACAUCUCCGCGUGUAUCGACAUCAAUGAAUGUGAGCUGAGUGCACACCUGUGCCCCAAUGGCCGUUGCGUGAACCUCAUAGGGAAGUAUCAGUGUGCCUGCAACCCUGGCUACCAUUCAACUCCCGAUAGGCUAUUUUGUGUUGACAUUGAUGAAUGCAGCAUAAUGAAUGGUGGUUGUGAAACCUUCUGCACAAACUCUGAAGGCAGCUAUGAAUGUAGCUGUCAGCCGGGAUUUGCACUAAUGCCUGACCAGAGAUCAUGCACCGACAUCGAUGAGUGUGAAGAUAAUCCCAAUAUCUGUGAUGGUGGUCAGUGCACAAAUAUCCCUGGAGAGUACAGGUGCUUGUGUUAUGAUGGAUUCAUGGCAUCUGAAGACAUGAAGACUUGUGUAGAUGUCAAUGAGUGUGACCUGAAUCCAAAUAUCUGCCUAAGUGGGACCUGUGAAAACACGAAAGGCUCAUUUAUCUGCCACUGUGAUAUGGGCUACUCCGGCAAAAAAGGAAAAACUGGCUGUACAGACAUCAAUGAAUGUGAAAUUGGAGCACACAACUGUGGCAAACAUGCUGUAUGUACCAAUACAGCAGGAAGCUUCAAAUGUAGCUGCAGUCCCGGGUGGAUUGGAGAUGGCAUUAAGUGCACUGAUCUGGACGAAUGUUCCAAUGGAACCCAUAUGUGCAGCCAGCAUGCAGACUGCAAGAAUACCAUGGGAUCUUACCGCUGUCUGUGCAAGGAAGGAUACACAGGUGAUGGCUUCACUUGUACAGACCUUGAUGAGUGCUCUGAGAACCUGAAUCUCUGUGGCAAUGGCCAGUGCCUCAAUGCACCAGGAGGAUACCGCUGUGAAUGCGACAUGGGCUUCGUGCCCAGUGCUGACGGGAAAGCCUGUGAAGAUAUUGAUGAGUGCUCCCUUCCGAACAUCUGUGUCUUUGGAACUUGCCACAACCUCCCUGGCCUGUUCCGCUGUGAGUGUGAGAUAGGCUACGAACUGGACAGAAGCGGCGGGAACUGCACAGAUGUGAAUGAAUGCCUGGAUCCAACCACGUGCAUCAGUGGGAACUGUGUCAACACUCCAGGCAGCUAUAUCUGUGACUGCCCACCUGAUUUUGAACUGAACCCAACUCGAGUUGGCUGUGUUGAUACCCGCUCUGGAAAUUGCUAUUUGGAUAUUCGACCUCGAGGAGACAAUGGAGAUACAGCCUGCAGCAAUGAAAUUGGAGUUGGUGUUUCCAAAGCUUCCUGCUGCUGUUCUCUGGGUAAAGCCUGGGGUACUCCUUGUGAGAUGUGUCCUGCUGUGAACACAUCCGAGUACAAAAUUCUUUGUCCUGGAGGGGAAGGUUUCCGACCAAAUCCUAUCACCGUUAUAUUGGAAGAUAUUGAUGAGUGCCAGGAGCUACCAGGGCUGUGCCAAGGAGGAAAAUGUAUCAACACCUUUGGGAGUUUCCAGUGCCGCUGUCCAACCGGCUACUACCUGAAUGAAGAUACACGAGUGUGUGAUGAUGUGAAUGAAUGUGAGACUCCUGGAAUCUGUGGUCCAGGGACAUGUUACAACACCGUUGGCAACUACACCUGUAUCUGUCCUCCAGACUACAUGCAAGUGAAUGGGGGAAAUAAUUGCAUGGAUAUGAGAAGAAGUUUGUGCUACAGAAACUACUAUGCUGACAACCAGACCUGUGAUGGAGAAUUGUUAUUCAACAUGACCAAGAAGAUGUGCUGCUGUUCCUACAACAUUGGCCGGGCGUGGAACAAGCCCUGUGAACAGUGUCCCAUCCCAAGUACAGAUGAGUUUGCUACACUCUGUGGAAGUCAAAGGCCAGGCUUUGUCAUCGACAUUUAUACCGGUUUACCCGUUGAUAUUGAUGAGUGCCGGGAGAUCCCAGGGGUCUGUGAAAAUGGAGUGUGUAUCAACAUGGUUGGCAGCUUCCGAUGUGAAUGUCCAGUGGGAUUCUUCUAUAAUGACAAGUUGUUGGUUUGUGAAGAUAUUGACGAGUGUCAGAACGGCCCAGUGUGCCAGCGCAACGCCGAAUGCAUCAACACUGCAGGCAGCUACCGCUGUGACUGUAAGCCCGGCUACCGCUUCACCUCCACAGGACAGUGCAAUGAUCGUAAUGAAUGUCAAGAAAUCCCCAAUAUAUGCAGUCAUGGGCAGUGCAUUGACACAGUUGGAAGCUUUUAUUGCCUUUGCCACACUGGUUUUAAAACAAAUGAUGACCAAACCAUGUGCUUGGACAUAAAUGAAUGUGAAAGAGAUGCCUGUGGGAAUGGAACUUGCCGGAACACAAUUGGUUCCUUCAACUGCCGCUGCAAUCAUGGUUUCAUCCUUUCUCACAACAAUGACUGUAUAGAUGUUGAUGAAUGUGCAAGUGGAAAUGGGAAUCUUUGCAGAAAUGGCCAAUGCAUUAAUACAGUGGGGUCUUUCCAGUGCCAGUGCAAUGAAGGCUAUGAGGUGGCUCCAGAUGGGAGGACCUGUGUGGAUAUCAAUGAAUGUCUUCUAGAACCCAGAAAAUGUGCACCAGGUACCUGUCAAAACUUGGAUGGGUCCUACAGAUGCAUUUGCCCACCUGGAUACAGUCUUCAAAAUGAGAAGUGUGAAGAUAUUGAUGAGUGUGUCGAAGAGCCAGAAAUUUGUGCCCUGGGCACAUGCAGUAACACUGAAGGCAGCUUCAAAUGUCUGUGUCCAGAAGGGUUUUCCUUGUCCUCCAGUGGAAGAAGGUGCCAAGAUUUGCGAAUGAGCUACUGUUAUGCGAAGUUUGAAGGAGGAAAGUGUUCAUCACCCAAAUCCAGAAAUCACUCCAAGCAGGAAUGCUGCUGUGCCUUGAAGGGAGAAGGCUGGGGAGACCCCUGCGAGCUCUGCCCCACGGAACCUGAUGAGGCCUUCCGCCAGAUAUGUCCUUAUGGAAGUGGGAUCAUCGUGGGACCUGAUGAUUCAGCAGUUGAUAUGGACGAAUGCAAAGAACCCGAUGUCUGUAAACAUGGACAGUGCAUCAAUACAGAUGGUUCCUAUCGCUGCGAGUGUCCCUUUGGUUAUAUUCUAGCAGGGAAUGAAUGUGUAGAUACUGAUGAAUGUUCUGUUGGCAAUCCUUGUGGAAAUGGAACCUGCAAGAAUGUGAUUGGAGGUUUUGAAUGCACCUGCGAGGAGGGAUUUGAGCCCGGUCCAAUGAUGACAUGUGAAGAUAUAAAUGAAUGUGCCCAGAAUCCUCUGCUCUGUGCCUUCCGAUGUGUGAACACUUAUGGGUCAUAUGAAUGCAAAUGUCCCGUGGGAUAUGUGCUCAGAGAAGACCGUAGGAUGUGCAAAGAUGAGGAUGAGUGUGAAGAGGGAAAACAUGACUGUACUGAAAAACAAAUGGAAUGCAAGAACCUCAUUGGCACAUAUAUGUGCAUCUGUGGACCCGGGUAUCAGCGGAGACCUGAUGGAGAAGGCUGUGUAGAUGAGAAUGAAUGUCAGACGAAGCCAGGGAUCUGUGAGAAUGGGCGCUGCCUCAACACCCGUGGGAGCUACACCUGUGAGUGUAAUGAUGGGUUUACCGCCAGCCCCAACCAGGACGAGUGCCUUGACAAUCGGGAAGGGUACUGCUUCACAGAGGUGCUACAAAACAUGUGUCAGAUCGGCUCCAGCAACAGGAACCCCGUCACCAAAUCGGAAUGCUGCUGUGACGGAGGGAGAGGCUGGGGUCCCCACUGUGAGAUCUGCCCUUUCCAGGGGACUGUGGCUUUCAAGAAACUCUGUCCCCAUGGCCGAGGAUUCAUGACCAAUGGAGCAGAUAUCGAUGAAUGCAAGGUUAUUCACGAUGUUUGCCGAAAUGGGGAAUGUGUCAAUGACAGAGGAUCAUAUCAUUGCAUUUGUAAAACUGGGUACACUCCAGAUAUAACUGGGACUUCCUGUGUAGAUCUGAACGAGUGCAACCAGGCUCCCAAACCCUGCAAUUUUAUCUGCAAAAACACAGAAGGGAGUUACCAGUGUUCAUGCCCGAAAGGCUACAUUCUGCAAGAGGAUGGAAGGAGCUGCAAAGAUCUUGAUGAGUGUGCAACCAAGCAACACAACUGCCAGUUCCUAUGUGUUAACACCAUUGGCGGCUUCACAUGCAAAUGUCCUCCCGGAUUUACCCAACACCAUACGUCCUGCAUUGAUAACAAUGAAUGCACCUCUGACAUCAAUCUGUGCGGGUCUAAGGGCAUUUGCCAGAACACUCCUGGAAGCUUCACCUGUGAAUGCCAGCGGGGAUUCUCACUUGAUCAGACCGGCUCCAGCUGUGAAGACGUGGACGAGUGUGAGGGUAACCACCGCUGCCAGCAUGGCUGCCAGAACAUCAUUGGGGGCUACAGGUGCAGCUGCCCCCAGGGCUACCUCCAGCACUACCAGUGGAACCAGUGUGUUGAUGAAAACGAAUGCCUCAGCGCUCACAUCUGCGGAGGAGCCUCCUGUCACAACACCCUGGGGAGCUACAAGUGCAUGUGUCCCGCCGGCUUCCAGUAUGAACAGUUCAGUGGAGGAUGCCAAGACAUCAAUGAAUGUGGCUCUGCGCAGGCCCCCUGCAGCUAUGGCUGUUCCAAUACCGAGGGCGGUUACCUGUGUGGCUGUCCACCUGGUUACUUCCGCAUAGGCCAAGGGCACUGUGUUUCUGGAAUGGGCAUGGGCCGAGGAAACCCAGAGCCACCUGUCAGUGGUGAAAUGGAUGACAAUUCACUCUCCCCAGAGGCUUGUUACGAGUGUAAGAUCAAUGGCUACCCCAAACGGGGCAGGAAACGGAGAAGCACAAACGAAACUGAUGCCUCCAAUAUCGAGGAUCAGUCUGAGACAGAAGCCAAUGUGAGUCUUGCAAGUUGGGAUGUUGAGAAGACAGCCAUCUUUGCUUUCAAUAUUUCCCACGUCAGUAACAAGGUUCGAAUCCUAGAACUCCUUCCAGCUCUUACAACUCUGACGAAUCACAACAGAUACUUGAUCGAAUCUGGAAAUGAAGAUGGCUUCUUUAAAAUCAACCAAAAGGAAGGGAUCAGCUACCUCCACUUCACAAAGAAGAAGCCAGUGGCUGGAACCUAUUCAUUACAAAUCAGUAGUACUCCACUUUAUAAAAAGAAAGAACUUAACCAACUAGAAGACAAAUAUGACAAAGACUACCUCAGUGGUGAACUGGGUGAUAAUCUGAAGAUGAAAAUCCAGGUUUUGCUUCAUUAAAUGCGUCGAGGGCGUCUGCUGGAGAUCGCCCUGGGAUUUACCGUGCUUUUAGCGUCCUACACGAGCCAUGGGGCGGACGCCAAUUUGGAGGCUGGGAACGUGAAGGAAACCAGAGCCAGUCGGGCCAAGAGAAGAGGCGGUGGAGGACACGACGCGCUUAAAGGACCCAAUGUCUGUGGAUCACGUUAUAAUGCUUACUGUUGCCCUGGAUGGAAAACCUUACCUGGCGGAAAUCAGUGUAUUGUCCCCAUUUGCCGGCAUUCCUGUGGGGAUGGAUUUUGUUCGAGGCCAAAUAUGUGCACUUGCCCAUCUGGUCAGAUAGCUCCUUCCUGUGGCUCCAGAUCCAUACAACACUGCAAUAUUCGCUGUAUGAAUGGAGGUAGCUGCAGUGACGAUCACUGUCUAUGCCAGAAAGGAUACAUAGGGACUCACUGUGGACAACCUGUUUGUGAAAGUGGCUGUCUCAAUGGAGGAAGGUGUGUGGCCCCAAAUCGAUGUGCAUGCACUUACGGAUUUACUGGACCCCAGUGUGAAAGAGAUUACAGGACAGGCCCAUGUUUUACUGUGAUCAGCAACCAGAUGUGCCAGGGACAACUCAGCGGGAUUGUCUGCACAAAAACGCUCUGCUGUGCCACAGUCGGCCGAGCCUGGGGCCACCCCUGUGAGAUGUGUCCUGCCCAGCCUCACCCCUGCCGCCGUGGCUUCAUUCCAAAUAUCCGCACGGGAGCUUGUCAAGAUGUGGAUGAAUGCCAGGCCAUCCCCGGGCUCUGUCAGGGAGGAAAUUGCAUUAAUACUGUUGGGUCUUUUGAGUGCAAAUGCCCUGCUGGACACAAACUUAAUGAAGUGUCACAAAAAUGUGAAGUUCCAGAAUUCAGACUCCAUUUUCAAGUUGUGAUCGGGCGGCGGAUGCAGCAUGAAUAAAUGCGUCGAGGGCGUCUGCUGGAGAUCGCCCUGGGAUUUACCGUGCUUUUAGCGUCCUACACGAGCCAUGGGGCGGACGCCAAUUUGGAGGCUGGGAACGUGAAGGAAACCAGAGCCAGUCGGGCCAAGAGAAGAGGCGGUGGAGGACACGACGCGCUUAAAGGGAUUGCUCCUGUAUAUAAAGGACGUGGGGAUUUAAUUCUGUGGCUGAAGAUCUGCAGAGAGGUGUGA



```python

```

## Translating RNA into Proteins

```python
# Prompt the user to enter the input RNA file name

input_file_name = input("Enter the name of the input RNA file:")
```

    Enter the name of the input RNA file: Fibrillin_RNA.txt



```python
# Open the input RNA file and read the RNA sequence

with open(input_file_name, "r") as input_file:
    rna_sequence = input_file.read().strip()
```


```python
# Define the codon table

codon_table = {# Phenylalanine (F)
    'UUU': 'F', 'UUC': 'F',
    # Leucine (L)
    'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
    # Isoleucine (I)
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
    # Methionine (M) - Start codon
    'AUG': 'M',
    # Valine (V)
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
    # Serine (S)
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',
    # Proline (P)
    'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    # Threonine (T)
    'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    # Alanine (A)
    'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    # Tyrosine (Y)
    'UAU': 'Y', 'UAC': 'Y',
    # Histidine (H)
    'CAU': 'H', 'CAC': 'H',
    # Glutamine (Q)
    'CAA': 'Q', 'CAG': 'Q',
    # Asparagine (N)
    'AAU': 'N', 'AAC': 'N',
    # Lysine (K)
    'AAA': 'K', 'AAG': 'K',
    # Aspartic Acid (D)
    'GAU': 'D', 'GAC': 'D',
    # Glutamic Acid (E)
    'GAA': 'E', 'GAG': 'E',
    # Cysteine (C)
    'UGU': 'C', 'UGC': 'C',
    # Tryptophan (W)
    'UGG': 'W',
    # Arginine (R)
    'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    # Glycine (G)
    'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    # Stop codons
    'UAA': '*', 'UAG': '*', 'UGA': '*'
}

```


```python
# Translate RNA to protein

protein_sequence = " "
for i in range(0, len(rna_sequence),3):
    codon = rna_sequence[i:i+3]
    if len(codon) == 3:
        amino_acid = codon_table[codon]
        if amino_acid == "*":
            break
        protein_sequence += amino_acid
```


```python
# Prompt the user to enter the output file name

output_file_name = input("Enter the name of the output file:")
```

    Enter the name of the output file: Fillibrin_Protein.txt



```python
# Save the protein sequence to a text file

with open(output_file_name,"w") as output_file:
    output_file.write(protein_sequence)
    print(f"The protein sequence has been saved to {output_file_name}")
```

    The protein sequence has been saved to Fillibrin_Protein.txt



```python
print(protein_sequence)
```

     MRRGRLLEIALGFTVLLASYTSHGADANLEAGNVKETRASRAKRRGGGGHDALKGPNVCGSRYNAYCCPGWKTLPGGNQCIVPICRHSCGDGFCSRPNMCTCPSGQIAPSCGSRSIQHCNIRCMNGGSCSDDHCLCQKGYIGTHCGQPVCESGCLNGGRCVAPNRCACTYGFTGPQCERDYRTGPCFTVISNQMCQGQLSGIVCTKTLCCATVGRAWGHPCEMCPAQPHPCRRGFIPNIRTGACQDVDECQAIPGLCQGGNCINTVGSFECKCPAGHKLNEVSQKCEDIDECSTIPGICEGGECTNTVSSYFCKCPPGFYTSPDGTRCIDVRPGYCYTALTNGRCSNQLPQSITKMQCCCDAGRCWSPGVTVAPEMCPIRATEDFNKLCSVPMVIPGRPEYPPPPLGPIPPVLPVPPGFPPGPQIPVPRPPVEYLYPSREPPRVLPVNVTDYCQLVRYLCQNGRCIPTPGSYRCECNKGFQLDLRGECIDVDECEKNPCAGGECINNQGSYTCQCRAGYQSTLTRTECRDIDECLQNGRICNNGRCINTDGSFHCVCNAGFHVTRDGKNCEDMDECSIRNMCLNGMCINEDGSFKCICKPGFQLASDGRYCKDINECETPGICMNGRCVNTDGSYRCECFPGLAVGLDGRVCVDTHMRSTCYGGYKRGQCIKPLFGAVTKSECCCASTEYAFGEPCQPCPAQNSAEYQALCSSGPGMTSAGSDINECALDPDICPNGICENLRGTYKCICNSGYEVDSTGKNCVDINECVLNSLLCDNGQCRNTPGSFVCTCPKGFIYKPDLKTCEDIDECESSPCINGVCKNSPGSFICECSSESTLDPTKTICIETIKGTCWQTVIDGRCEININGATLKSQCCSSLGAAWGSPCTLCQVDPICGKGYSRIKGTQCEDIDECEVFPGVCKNGLCVNTRGSFKCQCPSGMTLDATGRICLDIRLETCFLRYEDEECTLPIAGRHRMDACCCSVGAAWGTEECEECPMRNTPEYEELCPRGPGFATKEITNGKPFFKDINECKMIPSLCTHGKCRNTIGSFKCRCDSGFALDSEERNCTDIDECRISPDLCGRGQCVNTPGDFECKCDEGYESGFMMMKNCMDIDECQRDPLLCRGGVCHNTEGSYRCECPPGHQLSPNISACIDINECELSAHLCPNGRCVNLIGKYQCACNPGYHSTPDRLFCVDIDECSIMNGGCETFCTNSEGSYECSCQPGFALMPDQRSCTDIDECEDNPNICDGGQCTNIPGEYRCLCYDGFMASEDMKTCVDVNECDLNPNICLSGTCENTKGSFICHCDMGYSGKKGKTGCTDINECEIGAHNCGKHAVCTNTAGSFKCSCSPGWIGDGIKCTDLDECSNGTHMCSQHADCKNTMGSYRCLCKEGYTGDGFTCTDLDECSENLNLCGNGQCLNAPGGYRCECDMGFVPSADGKACEDIDECSLPNICVFGTCHNLPGLFRCECEIGYELDRSGGNCTDVNECLDPTTCISGNCVNTPGSYICDCPPDFELNPTRVGCVDTRSGNCYLDIRPRGDNGDTACSNEIGVGVSKASCCCSLGKAWGTPCEMCPAVNTSEYKILCPGGEGFRPNPITVILEDIDECQELPGLCQGGKCINTFGSFQCRCPTGYYLNEDTRVCDDVNECETPGICGPGTCYNTVGNYTCICPPDYMQVNGGNNCMDMRRSLCYRNYYADNQTCDGELLFNMTKKMCCCSYNIGRAWNKPCEQCPIPSTDEFATLCGSQRPGFVIDIYTGLPVDIDECREIPGVCENGVCINMVGSFRCECPVGFFYNDKLLVCEDIDECQNGPVCQRNAECINTAGSYRCDCKPGYRFTSTGQCNDRNECQEIPNICSHGQCIDTVGSFYCLCHTGFKTNDDQTMCLDINECERDACGNGTCRNTIGSFNCRCNHGFILSHNNDCIDVDECASGNGNLCRNGQCINTVGSFQCQCNEGYEVAPDGRTCVDINECLLEPRKCAPGTCQNLDGSYRCICPPGYSLQNEKCEDIDECVEEPEICALGTCSNTEGSFKCLCPEGFSLSSSGRRCQDLRMSYCYAKFEGGKCSSPKSRNHSKQECCCALKGEGWGDPCELCPTEPDEAFRQICPYGSGIIVGPDDSAVDMDECKEPDVCKHGQCINTDGSYRCECPFGYILAGNECVDTDECSVGNPCGNGTCKNVIGGFECTCEEGFEPGPMMTCEDINECAQNPLLCAFRCVNTYGSYECKCPVGYVLREDRRMCKDEDECEEGKHDCTEKQMECKNLIGTYMCICGPGYQRRPDGEGCVDENECQTKPGICENGRCLNTRGSYTCECNDGFTASPNQDECLDNREGYCFTEVLQNMCQIGSSNRNPVTKSECCCDGGRGWGPHCEICPFQGTVAFKKLCPHGRGFMTNGADIDECKVIHDVCRNGECVNDRGSYHCICKTGYTPDITGTSCVDLNECNQAPKPCNFICKNTEGSYQCSCPKGYILQEDGRSCKDLDECATKQHNCQFLCVNTIGGFTCKCPPGFTQHHTSCIDNNECTSDINLCGSKGICQNTPGSFTCECQRGFSLDQTGSSCEDVDECEGNHRCQHGCQNIIGGYRCSCPQGYLQHYQWNQCVDENECLSAHICGGASCHNTLGSYKCMCPAGFQYEQFSGGCQDINECGSAQAPCSYGCSNTEGGYLCGCPPGYFRIGQGHCVSGMGMGRGNPEPPVSGEMDDNSLSPEACYECKINGYPKRGRKRRSTNETDASNIEDQSETEANVSLASWDVEKTAIFAFNISHVSNKVRILELLPALTTLTNHNRYLIESGNEDGFFKINQKEGISYLHFTKKKPVAGTYSLQISSTPLYKKKELNQLEDKYDKDYLSGELGDNLKMKIQVLLH



```python

```


