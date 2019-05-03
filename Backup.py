%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
import seaborn as sns

# read csv file
beijing = pd.read_csv('new.csv', encoding = 'unicode_escape', low_memory=False)


### A Glance
**Context** Housing price of Beijing from *2011* to *2017*, fetching from Lianjia.com


# take a look at the data
beijing.head(5)
beijing.shape
beijing.info()
beijing.describe()

# drop url column since the id column is unique identifier for each instance
print(beijing.url.nunique() == beijing.id.nunique() == beijing.shape[0])
df = beijing.drop(['url'], axis=1)
# reset id column as index of the table
df.set_index(['id'], inplace=True)
# take a look at the column names now
print(df.columns)

# look for columns containing NaN type information
print(df.isna().any())

# we are going to take a look at these columns (with DOM info)
temp1 = df[['DOM', 'buildingType', 'elevator', 'fiveYearsProperty', 'subway', 'communityAverage']]
temp1_null = temp1[pd.isnull(temp1).any(axis=1)]
temp1_null.shape

print('column name: (# of nulls in the temp1 df,  # of nulls in the original df)')
print('buildingType: (', temp1_null.buildingType.isnull().sum(), ', ', 318851 - 316830, ')')
print('elevator: (', temp1_null.elevator.isnull().sum(), ', ', 318851 - 318819, ')')
print('fiveYearsProperty: (', temp1_null.fiveYearsProperty.isnull().sum(), ', ', 318851 - 318819, ')')
print('subway: (', temp1_null.subway.isnull().sum(), ', ', 318851 - 318819, ')')
print('communityAverage: (', temp1_null.communityAverage.isnull().sum(), ', ', 318851 - 318388, ')')

# columns containing NaN values (without DOM info)
temp2 = df[['buildingType', 'elevator', 'fiveYearsProperty', 'subway', 'communityAverage']]
temp2_null = temp2[pd.isnull(temp2).any(axis=1)]
temp2_null.shape

print('column name: (# of nulls in the temp1 df,  # of nulls in the original df)')
print('buildingType: (', temp2_null.buildingType.isnull().sum(), ', ', 318851 - 316830, ')')
print('elevator: (', temp2_null.elevator.isnull().sum(), ', ', 318851 - 318819, ')')
print('fiveYearsProperty: (', temp2_null.fiveYearsProperty.isnull().sum(), ', ', 318851 - 318819, ')')
print('subway: (', temp2_null.subway.isnull().sum(), ', ', 318851 - 318819, ')')
print('communityAverage: (', temp2_null.communityAverage.isnull().sum(), ', ', 318851 - 318388, ')')

temp2_null.describe()

# columns containing NaN values (without DOM info)
temp3 = df[['buildingType', 'elevator', 'fiveYearsProperty', 'subway', 'communityAverage']]
temp3_null = temp3[pd.isnull(temp3[['elevator', 'fiveYearsProperty', 'subway']]).any(axis=1)]
temp3_null


#### What we got so far:
- According to Pandas' document, **The 50 percentile is the same as the median**. That is, if we look at the `temp2_null.describe()` above, the distribution of the '*communityAverage*' column is skewed to the right (mean > median); in the original data set, the mean of the communityAverage is 63682.446305 and the median is 59015.000000, meaning that in the original data set, the distribution of the communityAverage is also skewed to the right. 
- The instances containing the NaN values (regardless the DOM column) mainly fall on the left side of the distribution of the original data set respect to the communityAverage column. One potential reason why these instances fall in the lower communityAverate price zone is that these properties are located in the suburb area. We will verify this hypothesis later when we make the lng/lat agains the total price and the community price. 
- Subway explanation is not provided by the uploader of the data set. I selected several instances and went into the original pages tried to find some information. The instances with NaN values in the subway column that I looked, there were no information about the subway condition, and on the map these properties are far away from major public transportations. The instances with value 1 in the subway column generally have subway station nearby and the distances are less than 1000m from the properties to nearest subway station. Those has value 0 in the subway column, in the original webpage, they ether have no information about the nearby subway or the nearest subway station is more than 1000m away.Consider there are only 32 instances containing NaN value in the subway column. I'm considering either set these value to 0 or simply remove them from the dataset.
- In `temp3_null` we can see that the instances with no *elevator, fiveYearsProperty, subway* informations are the same instances. Interesting thing is that the buildingType column, these 32 instances has value less than 1. Which is bizarre, because in the column explanation: buildingType: including tower( 1 ) , bungalow( 2 )，combination of plate and tower( 3 ), plate( 4 ). I think these data are bad data. since 32 is a very small number compare to 318851, I believe it is safe to remove these instances from the original dataset.


### Data Cleaning


# as promised previously, i will drop the instances with buildingType value smaller than 1.
df = df.dropna(axis=0, subset=['elevator'])
df.shape

# now let's take a look at the geographical information of the data.
# because the original dataset is way to large, I will sample some instances to plot it, n=10000
smpl = df.sample(n=10000, random_state=1)

smpl.plot(kind="scatter", x="Lng", y='Lat', alpha=0.2, s=(smpl['square']/3)**1.3, label='square', c='price', cmap=plt.get_cmap('hot'), colorbar=True, figsize=(15, 12))
plt.title('Sample Home Price Map')
plt.legend()


#### There are many things going on in the image:
1. each circle is a property instances and the location on the axis represents the actual location of the property on the map (with the longitude and latitude information).
2. the size of the circle represents the size (square meters) of the property sold.
3. the heatmap is showing the price per square meter of the property. The lighter the color is, the higher price per square meter the property is.
4. the empty area in the middle of the image is the "heart" of Beijing, which is where the government facilities located (that's why no homes for sale around the area). 
5. notice the top part of the government owned area is slightly more price (yellower) than the bottom part. This is because those two areas are the district of finance and commerce. 
6. the 'spikes' are subway lines. The further away from the center of beijing, the lower the price per square meter. also notice that the circle sizes near the subway lines are generally smaller than the other areas. This should be because there are mainly apartment complex in the area, which these home owners are mainly relying on public transportation when they need to go out.'
7. there are some pretty large properties located near the "heart" of Beijing. I'm not surprised because there are some really wealthy people living there since Beijing is the capital of China. 
8. on the north side of the circle, there is an very dark area has many properties and relatively low per square meter price. This area is near the subway line 13 and 15. These two lines were built after 2010. It is less convenient for living (less shops and other entertainment places). I lived near the area shortly before. 


# here is a map for Beijing subway as an illustration of the location associate with the image we plotted previously.
figure(num=None, figsize=(12, 12), dpi=80)
img=mpimg.imread('subway.jpg')
imgplot = plt.imshow(img, aspect='equal')
plt.show()

# I viewed the data set in Excel and realized in some columns the data contains Chinese charactors, 
# which after writing into Pandas DataFrame they become some random charactors. 
# for example in the 'floor' column, there are characters "高，中，低" meaning "High, Mid, Low". 
# It follows by the exact floor number. 
# I believe I can safely remove these Chinese characters without loosing information of the data.
df['floor'] = df.floor.str.replace('^[^\d]*', '').astype(float)

# change livingRoom and drawingRoom columns from object type to float type
df['livingRoom'] = df.livingRoom.astype(float)
df['drawingRoom'] = df.drawingRoom.astype(float)
df['bathRoom'] = df.bathRoom.astype(float)

# now we have tradeTime and constructionTime columns are object type,
# since these are time series information, we will leave them as is for now.
# also the DOM, buildingType, and communityAverage columns still have null values,
# we will fix them now.
temp4 = df[['DOM', 'buildingType', 'communityAverage', 'price', 'totalPrice', 'square', 'livingRoom', 'drawingRoom', 'bathRoom', 'renovationCondition', 'district']]
building_null = temp4[pd.isnull(temp4[['buildingType']]).any(axis=1)]
community_null = temp4[pd.isnull(temp4[['communityAverage']]).any(axis=1)]
print(building_null.shape, community_null.shape)

# notice there are two significant outliers here. Let's take a look at these two guys.
sns.pairplot(building_null[['price', 'totalPrice', 'square', 'district', 'bathRoom']])

# Let's take a look at the two significant outliers
building_null[building_null['totalPrice'] >= 10000]

# I double checked the home information, these are luxury townhomes / houses. 
# they are located near the center area of beijing but are very large in size and high in prices.
print(beijing[beijing['id'] == '101101209445']['url'])
print(beijing[beijing['id'] == '101101263750']['url'])

# how the pairplot looks like without these two extreme instances
no_outliers = building_null[building_null['totalPrice'] < 10000]
sns.pairplot(no_outliers[['price', 'totalPrice', 'square', 'district', 'bathRoom']])

# from this pairplot we know that these properties without buildingType information fall into the lower price zone.
# majority of the properties are located in district 6 and district 7. 
# Let's take a look at how the price and square feet information look like in these two districts.
# from pairplot we can tell that in the subset of the data, district 7 has way higher total price
# But the price per square meter distributed relatively evenly.
# This is proved by the following plot, that district 7 is generally more pricy compare to district 6.
six = df[df.district == 6]
seven = df[df.district == 7]

six.plot(kind="scatter", x="Lng", y='Lat', alpha=0.2, s=(six['square']/5)**1.1, label='square', c='price', cmap=plt.get_cmap('hot'), colorbar=True, figsize=(7, 5))
plt.title('District 6')
seven.plot(kind="scatter", x="Lng", y='Lat', alpha=0.2, s=(seven['square']/5)**1.1, label='square', c='price', cmap=plt.get_cmap('hot'), colorbar=True, figsize=(7, 5))
plt.title('District 7')
plt.show()

# see how the price and totalPrice is distributed of the whole dataset
sns.distplot(beijing.price)
plt.title('Distribution of Price / Sqr Meter')
