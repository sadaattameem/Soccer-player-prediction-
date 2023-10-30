#!/usr/bin/env python
# coding: utf-8

# # Data Mining & Visualization Group Project EDA and Modeling
# 
# #### By Sadaat Tameem, Business Analyst

# ## EDA Outline:
# 1) Overall Dataframe
# 2) Output Variable
# 3) Continuous Variables
# 4) Categorical Variables
# 5) Notes 

# In[3]:


# Import Packages
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import numpy as np 
import seaborn as sns 
from pandas import Series, DataFrame 
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf


# In[4]:


#Read in dataframe
df_0=pd.read_csv("2022_Pro_Soccer_Data - All_Clubs.csv", header=1)
df_0.head()


# In[5]:


#set player as the index
df_0.index=list(df_0.Player)   
df_0.head(3)


# In[6]:


#delete Player Column
df = df_0.drop('Player',axis=1)
df.head(3)


# ### 1) Overall Dataframe

# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.corr()


# In[11]:


corr = df.corr()
kot = corr[corr>=.8]
plt.figure(figsize=(30,15))
sns.heatmap(kot, annot = True, cmap="Reds").set(title='Correlation Matrix Heat Map')


# In[12]:


#turning the correlation matrix into a dataframe
c = df.corr()
s = c.unstack()
so = s.sort_values()
corr_0 = DataFrame(data=so)
corr_0.head()


# In[13]:


#new column header
corrdf=corr_0.rename(columns={0:"Correlation"})
corrdf.head()


# In[14]:


#deleting rows where correlation is exactly 1, not useful
corrdf = corrdf[corrdf.Correlation != 1]
corrdf


# In[15]:


#show me the values where the correlation is greater than .7
highcorr=corrdf.loc[corrdf.Correlation>.7]
highcorr


# ##### Notes: 
# Correlations are going to be a problem.I don't have 290 high correlations, but I probably have 145 since everything in a correlation matrix is listed twice. 

# ### 2) Analyze Output Variable - Salary_Euro

# In[16]:


df.Salary_Euro.dtypes


# In[17]:


round(df['Salary_Euro'].describe(),2)


# In[18]:


#additional measures of central tendency

#median
print(f"The median of Salary in Euros is {round(df.Salary_Euro.median(),2)}")

#mode
import statistics
print(f'The mode of Salary in Euros is {statistics.mode(df.Salary_Euro)}.')


# In[19]:


# additional measures of spread

# range
print(f'The range of Salary in Euros is {round(df.Salary_Euro.max()-df.Salary_Euro.min(),2)}.')

# variance
print(f'The variance of Salary in Euros is {round(df.Salary_Euro.var(),2)}.')


# In[20]:


# distibution of the output variable
sns.displot(df.Salary_Euro)


# ### Other interesting views of Salary:

# In[21]:


# box and whisker plot
sns.boxplot(x="Salary_Euro", data=df, palette='colorblind').set(title='Box Plot of Salary in Millions of Euros')
plt.xlabel('Salary in Millions of Euros');


# In[22]:


plt.figure(figsize = (14,7))
sns.boxplot(x=df['Salary_Euro'], y=df['Club'], palette ='colorblind').set(title='Box Plot of Salary in Millions of Euros Per Club')
plt.xlabel('Salary in Millions of Euros')
plt.ylabel('Club');


# In[23]:


plt.figure(figsize = (14,5))
sns.boxplot(x=df['Salary_Euro'], y=df['Position_1'], palette ='colorblind').set(title='Box Plot of Salary in Millions of Euros Per Position')
plt.xlabel('Salary in Euros')
plt.ylabel('Main Position');


# In[24]:


HighestPaid=df.Salary_Euro.nlargest(n=10)
LowestPaid=df.Salary_Euro.nsmallest(n=10)
topandbottom10=pd.concat([HighestPaid,LowestPaid])

plt.rcParams["figure.figsize"] = (14, 5)
topandbottom10.plot(kind='bar', title='10 Highest and 10 Lowest Paid Players in Millions of Euros')
plt.xticks(rotation=25)
plt.ylabel('Salary in Millions of Euros')
plt.xlabel('Player');


# In[25]:


#Top 10 Paid Players
HighestPaid=df.Salary_Euro.nlargest(n=10)
HighestPaid.plot(kind='bar', title='Top 10 Highest Paid Players in Millions of Euros')
plt.xticks(rotation=25)
plt.ylabel('Salary in Millions of Euros')
plt.xlabel('Player');


# In[26]:


#Bottom 10 Paid Players
LowestPaid=df.Salary_Euro.nsmallest(n=10)
LowestPaid.plot(kind='bar', title='Top 10 Lowest Paid Players in Euros')
plt.xticks(rotation=25)
plt.ylabel('Salary in Euros')
plt.xlabel('Player');


# ### 3) Analyze Continuous Variables

# In[27]:


df.describe()


# In[28]:


#Distribution Graphs For Some Of The Output Variables

plt.figure(figsize=(7,30)) #(width,height)
plt.subplots_adjust(hspace = .8)

#Age
plt.subplot(9, 1, 1)
plt.hist(df.Age,15) 
plt.title('Age',fontsize='10')
plt.xticks(rotation=45);

#Matches_Played
plt.subplot(9, 1, 2)
plt.hist(df.Matches_Played,15) 
plt.title('Number of Matches Played',fontsize='10')
plt.xticks(rotation=45);

#Goals
plt.subplot(9, 1, 3)
plt.hist(df.Goals,15) 
plt.title('Goals',fontsize='10')
plt.xticks(rotation=45);

#Assists
plt.subplot(9, 1, 4)
plt.hist(df.Assists,15) 
plt.title('Assists',fontsize='10')
plt.xticks(rotation=45);

#Passes_Completed 
plt.subplot(9, 1, 5)
plt.hist(df.Passes_Completed,15) 
plt.title('Passes_Completed',fontsize='10')
plt.xticks(rotation=45);

#Shots_Attempted
plt.subplot(9, 1, 6)
plt.hist(df.Shots_Attempted,15) 
plt.title('Shots_Attempted',fontsize='10')
plt.xticks(rotation=45);

#Progressive_Carrying_Distance_Yards
plt.subplot(9, 1, 7)
plt.hist(df.Progressive_Carrying_Distance_Yards,15) 
plt.title('Progressive_Carrying_Distance_Yards',fontsize='10')
plt.xticks(rotation=45);


# ##### Notes:
# I didn't make histograms of every continuous variable since I have so many, but I listed out a few above. Since I have so many 0s for players it makes all the histograms look off.I will have to address this in our modeling process. 

# In[29]:


#Scatter plots
salaryyellow=np.corrcoef(df.Salary_Euro,df.Yellow_Cards)[0][1]
df.plot.scatter('Salary_Euro','Yellow_Cards',title=f"Correlation of Salary and Yellow Cards:{salaryyellow:.2f}" );


# ### 4) Analyze Categorical Variables

# In[30]:


#Looking at categorical variables
df[['Club','Governing_Country','Nation_Of_Player','Position_1','Position_2' ]].describe()


# In[31]:


Club = DataFrame(df.Club.value_counts())
Club.plot(kind='bar')
plt.xticks(rotation=45);


# In[32]:


Nation = DataFrame(df.Nation_Of_Player.value_counts())

plt.rcParams["figure.figsize"] = (14, 5)
Nation.plot(kind='bar', title='Counts of Nation of Player')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.xlabel('Nation of Player');


# In[35]:


position = DataFrame(df.Position_1.value_counts())
position.plot(kind='bar')
plt.xticks(rotation=45);


# In[33]:


position2 = DataFrame(df.Position_2.value_counts())
position2.plot(kind='bar')
plt.xticks(rotation=45);


# ### EDA Notes:
# * There are 204 rows, 47 columns
# * Some nulls in the column Position_2. I can delete this row and concern ourselves with Position_1. Alternatively, I can make it a y/n column and put a 'y' for everyone with a second position and a 'n' for people without. 
# * Output variable is not normally distributed. 
# * Lots of highly correlated data, will need to deal with this when creating models, maybe try some PCA analysis
# * Only 4 positions are used in the dataset, makes it nice for modeling and analyzing the data
# * Since input 0s for all the missing data, it makes all the histograms not normally distributed. This is because, for example, goalies don't typically score goals, so their shots on target is 0, but that doesn't mean that stat isn't right for analysis. This is the same for all positions, the stats that are important for their position are full, but the rest might be null or 0. Makes it hard for modeling. 
# * Paris has some seriously highly paid players, superstars. Maybe I want to take those out of our modeling set.

# ## PCA

# In[34]:


from sklearn.datasets import load_digits #Digits Dataset
from sklearn.preprocessing import StandardScaler #for standardizing our data
 
data = df
data.shape


# In[35]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Select all variables except non-numeric columns (object type)
numeric_variables = df.select_dtypes(include='number')

# Drop any missing values
numeric_variables.dropna(inplace=True)

# Standardize the numeric variables
scaler = StandardScaler()
numeric_variables_scaled = scaler.fit_transform(numeric_variables)

# Perform PCA
pca = PCA()
numeric_variables_pca = pca.fit_transform(numeric_variables_scaled)

# Create a DataFrame for the PCA results
pca_columns = ['PC{}'.format(i+1) for i in range(pca.n_components_)]
df_pca = pd.DataFrame(data=numeric_variables_pca, columns=pca_columns)

# Print the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
for i, ratio in enumerate(explained_variance_ratio):
    print(f'PC{i+1}: {ratio:.4f}')

# You can access the principal components using df_pca['PC1'], df_pca['PC2'], etc.


# In[36]:


# Print the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
for i, ratio in enumerate(explained_variance_ratio):
    print(f'PC{i+1}: {ratio:.4f}')


# In[37]:


# Calculate cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)

# Find the number of components that explain a certain threshold of variance (e.g., 80%)
n_components_threshold = np.argmax(cumulative_variance >= 0.8) + 1

print(f"Number of components explaining 80% variance: {n_components_threshold}")


# Based on the explained variance ratios for each principal component (PC), it appears that the majority of the variance is explained by the first few components. The cumulative explained variance plot shows that approximately 80% of the variance is explained by the first 5 components. Therefore, it is recommended to retain these 5 components and discard the remaining components.
# 
# By reducing the dimensionality of the data to these 5 components, I can capture the most important patterns and information while discarding less influential components. This can lead to a more efficient and interpretable model.
# 
# It is important to note that the decision to retain a certain number of components should be based on the trade-off between explained variance and the desired level of dimensionality reduction. In this case, retaining 5 components explains a significant portion of the variance while still reducing the dimensionality considerably.
# 
# Further analysis and modeling can be performed using these 5 components as input variables, allowing for more focused and interpretable analysis of the data.

# In[38]:


# Calculate the proportion of explained variance
explained_variance_ratio = pca.explained_variance_ratio_

# Plot the proportion of explained variance
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Proportion of Explained Variance')
plt.title('Proportion of Explained Variance vs. Number of Components')
plt.grid(True)
plt.show()


# In[39]:


# Calculate the cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot the cumulative explained variance ratio
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio vs. Number of Components')
plt.grid(True)
plt.show()


# ## Linear Regression with Significant variables

# In[40]:


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Selecting the predictor variables and the target variable
predictors = ['Age', 'Goals', 'Assists', 'Red_Cards', 'Progressive_Passes', 'Progressive_Passes_Received',
              'Shots_Attempted', 'Shots_On_Target', 'Take_Ons_Attempted']
categorical_predictors = ['Club', 'Governing_Country']
target = 'Salary_Euro'

# Creating the predictor matrix X and the target vector y
X = df[predictors]
categorical_data = pd.get_dummies(df[categorical_predictors], drop_first=True)
X = pd.concat([X, categorical_data], axis=1)
y = df[target]

# Fit the OLS model
model = sm.OLS(y, X)
results = model.fit()

# Print the summary of the regression results
print(results.summary())


# The OLS regression results indicate that the overall model explains 64.4% of the variation in the dependent variable, Salary_Euro, as indicated by the R-squared value. The adjusted R-squared value, which accounts for the number of predictors in the model, is 61.4%.
# 
# Several predictor variables show statistically significant relationships with the salary. Age, Red_Cards, Progressive_Passes, Progressive_Passes_Received, Shots_Attempted, Shots_On_Target, Take_Ons_Attempted, Club_Arsenal, Club_Colorado Rapids, Club_Napoli, Club_Porto, Governing_Country_Germany, Governing_Country_Italy, Governing_Country_Netherlands, and Governing_Country_Portugal have coefficients with p-values less than 0.05, suggesting they have a significant impact on salary.
# 
# However, some variables such as Goals, Assists, Club_Barcelona, Club_Bayern Munich, and Club_Paris Saint-Germain are not statistically significant at the 0.05 level.
# 
# The regression model also indicates potential multicollinearity issues or a singular design matrix, as suggested by the condition number and the smallest eigenvalue.
# 
# Based on these results, it is recommended to consider the statistically significant predictors in determining player salaries. Variables such as age, red cards, progressive passes, shots attempted and on target, take ons attempted, and the club and governing country affiliations can be used as factors in salary negotiations. 
# 
# 
# Here is the interpretation of each variable's coefficient in relation to the target variable, Salary_Euro:
# 
# 1.Age: For every unit increase in Age, there is an increase of approximately 477,500 Euros in Salary_Euro, holding other variables constant. This suggests that older players tend to have higher salaries.
# 
# 2.Goals: For every unit increase in Goals, there is a decrease of approximately 636,200 Euros in Salary_Euro, holding other variables constant. This indicates that scoring more goals may not necessarily lead to higher salaries.
# 
# 3.Assists: For every unit increase in Assists, there is an increase of approximately 371,000 Euros in Salary_Euro, holding other variables constant. This suggests that players who provide more assists tend to have higher salaries.
# 
# 4.Red_Cards: For every unit increase in Red_Cards, there is an increase of approximately 4,390,000 Euros in Salary_Euro, holding other variables constant. This unexpected result implies that players who receive more red cards have higher salaries, which could be due to other factors related to their playing style or reputation.
# 
# 5.Progressive_Passes: For every unit increase in Progressive_Passes, there is an increase of approximately 23,500 Euros in Salary_Euro, holding other variables constant. This indicates that players who make more progressive passes tend to have higher salaries.
# 
# 6.Progressive_Passes_Received: For every unit increase in Progressive_Passes_Received, there is a decrease of approximately 28,600 Euros in Salary_Euro, holding other variables constant. This suggests that players who receive more progressive passes may have lower salaries, possibly because they are less involved in the attacking play.
# 
# 7.Shots_Attempted: For every unit increase in Shots_Attempted, there is a decrease of approximately 451,400 Euros in Salary_Euro, holding other variables constant. This indicates that players who attempt more shots may not necessarily command higher salaries.
# 
# 8.Shots_On_Target: For every unit increase in Shots_On_Target, there is an increase of approximately 1,541,000 Euros in Salary_Euro, holding other variables constant. This suggests that players who have more shots on target tend to have higher salaries.
# 
# 9.Take_Ons_Attempted: For every unit increase in Take_Ons_Attempted, there is an increase of approximately 90,300 Euros in Salary_Euro, holding other variables constant. This implies that players who attempt more take ons tend to have higher salaries.
# 
# 10.Club variables (e.g., Club_Arsenal, Club_Barcelona): The coefficients represent the salary difference compared to a reference club (likely omitted from the model). Positive coefficients indicate higher salaries compared to the reference club, while negative coefficients suggest lower salaries.
# 
# 11.Governing_Country variables (e.g., Governing_Country_Germany, Governing_Country_Italy): The coefficients represent the salary difference compared to a reference governing country (likely omitted from the model). Positive coefficients indicate higher salaries compared to the reference country, while negative coefficients suggest lower salaries.

# In[41]:


# import packages for splitting data and running linear regression
from sklearn.model_selection import train_test_split # For splitting the data into training/test datasets
import statsmodels.api as sm # For linear regression modeling
from statsmodels.compat import lzip # for additional capabilities like plotting
# splitting the data into training/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 123)


# In[42]:


# check to make sure the data is the correct shape 
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[43]:


#fit the model
model_results=model.fit()

# obtain our predicted values using our testing data
y_pred = model_results.predict(X_test)


# In[44]:


import numpy as np
import matplotlib.pyplot as plt

# Calculate the best-fit line
coefficients = np.polyfit(y_test, y_pred, 1)
trendline = np.poly1d(coefficients)

# Build a scatterplot
plt.scatter(y_test, y_pred)

# Add the trend line
plt.plot(y_test, trendline(y_test), color='red')

# Add a line for perfect correlation
plt.plot([x for x in range(1000, 150000)], [x for x in range(1000, 150000)], color='red')

# Label it nicely
plt.title("Predicted vs. Actual Values")
plt.xlabel("Actual")
plt.ylabel("Predicted")

# Display the plot
plt.show()


# Fit Plots
# 
# The plot_fit function plots the fitted values versus a chosen independent variable. It includes prediction confidence intervals and optionally plots the true dependent variable, which is shown below. fig = sm.graphics.plot_fit(prestige_model, "education") fig.tight_layout(pad=1.0)

# In[45]:


import statsmodels.api as sm

# Create and fit the OLS model
model = sm.OLS(y_train, X_train)
results = model.fit()

# Plot the fit
fig = sm.graphics.plot_fit(results, "Goals")
fig.tight_layout(pad=1.0)


# Single Variable Regression Diagnostics
# 
# The follownig plot uses the plot_regress_exog function. This returns a 2x2 plot containing the dependent variable and fitted values with confidence intervals vs. the independent variable chosen, the residuals of the model vs. the chosen independent variable, a partial regression plot, and a CCPR plot. This function can be used for quickly checking modeling assumptions with respect to a single regressor.

# In[54]:


ig = sm.graphics.plot_regress_exog(results, "Goals")
fig.tight_layout(pad=1.0)


# ## GLM

# In[46]:


df.hist(column="Goals")


# In[47]:


import statsmodels.api as sm

# Creating the design matrix X
X = df[['Age', 'Assists', 'Yellow_Cards','Red_Cards','Expected_Goals', 'Progressive_Carries', 'Touches']]
# Include more relevant independent variables from your dataset

# Adding a constant column to the design matrix
X = sm.add_constant(X)

# Selecting the response variable
y = df['Goals']

# Fitting the Poisson GLM
model = sm.GLM(y, X, family=sm.families.Poisson())
result = model.fit()

# Print the model summary
print(result.summary())


# In[48]:


# Obtaining the model's predicted values
y_pred = result.mu

# Calculating the residuals
residuals = y - y_pred

# Creating a residual plot
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Poisson GLM - Residual Plot')
plt.show()


# In[49]:


# Obtaining the standardized residuals
std_residuals = result.resid_pearson

# Creating a Q-Q plot
sm.qqplot(std_residuals, line='s')
plt.title('Poisson GLM - Q-Q Plot')
plt.show()


# In[50]:


# Creating a scatter plot of fitted values vs. actual values
plt.scatter(y_pred, y)
plt.plot([0, np.max(y_pred)], [0, np.max(y_pred)], color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Actual Values')
plt.title('Poisson GLM - Fitted vs. Actual')
plt.show()


# For the GLM model, I chose the Poisson regression method to predict the number of goals.
# 
# The variables I included in the model are:
# 
# Age: I included age as it is known to have an impact on a player's performance and goal-scoring ability. Younger players might have more energy and agility, which could contribute to higher goal-scoring rates.
# Assists: Assists can be a good indicator of a player's involvement in the attacking play and their ability to create goal-scoring opportunities for themselves and their teammates.
# Yellow_Cards: While yellow cards might seem unrelated to goal-scoring, they could be indicative of a player's aggression and competitiveness, which might influence their goal-scoring ability.
# Red_Cards: Similar to yellow cards, red cards could indicate a player's aggressiveness and could potentially impact their goal-scoring opportunities if they receive suspensions or bans.
# Expected_Goals: Expected goals is a metric that quantifies the quality of scoring opportunities a player has had. Including this variable helps capture the player's goal-scoring potential based on the quality of chances they have created or received.
# Progressive_Carries: Progressive carries measure the ability of a player to carry the ball forward and advance the attack. Players with higher progressive carry numbers might have a better chance of getting into goal-scoring positions.
# Touches: The number of touches a player has on the ball could reflect their involvement in the game and their ability to create scoring opportunities.
# The model results suggest that age, assists, expected goals, and touches have a significant impact on the number of goals scored. Younger players, those with more assists, higher expected goals, and more touches tend to score more goals.
# 
# Based on these findings, the following recommendations can be made:
# 
# Clubs should focus on recruiting or retaining younger players with high goal-scoring potential.
# Players with a high number of assists should be given more attention as they are likely to contribute to goal-scoring opportunities.
# Coaches should emphasize creating and converting high-quality goal-scoring chances, as indicated by the expected goals metric.
# Encouraging players to be more involved in the game and increase their number of touches could lead to increased goal-scoring opportunities.
# It's important to note that other variables not included in this model could also impact goal-scoring, such as playing position, playing time, and the quality of the opposition. Further analysis incorporating these variables could provide a more comprehensive understanding of goal-scoring in football.
# 
# Here are the odds ratios for the variables in the Poisson GLM model:
# 
# Age: The odds of scoring a goal decrease by a factor of exp(-0.0284) = 0.9719 for a one-year increase in age.
# Assists: The odds of scoring a goal increase by a factor of exp(0.0776) = 1.0808 for a one-unit increase in assists.
# Yellow_Cards: The odds of scoring a goal increase by a factor of exp(0.0378) = 1.0384 for a one-unit increase in yellow cards.
# Red_Cards: The odds of scoring a goal decrease by a factor of exp(-0.0920) = 0.9127 for a one-unit increase in red cards.
# Expected_Goals: The odds of scoring a goal increase by a factor of exp(0.1605) = 1.1746 for a one-unit increase in expected goals.
# Progressive_Carries: The odds of scoring a goal decrease by a factor of exp(-0.0006) = 0.9994 for a one-unit increase in progressive carries.
# Touches: The odds of scoring a goal increase by a factor of exp(0.0002) = 1.0002 for a one-unit increase in touches.

# ## LMM 

# In[51]:


import scipy.stats as stats
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")

# Ignore ConvergenceWarnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ignore ConvergenceWarnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


# In[52]:


#Read in dataframe
df_0=pd.read_csv("2022_Pro_Soccer_Data - All_Clubs.csv",header=1)
df_0.head()
pd.set_option('display.max_columns', None)


# In[53]:


df_0.index=list(df_0.Player)   
df_0.head(4)


# In[54]:


#delete Player Column
df = df_0.drop('Player',axis=1)
df.head(3)


# In[55]:


### 1) Analyze Overall Dataframe

### Linear Mixed Model

# In[5]:

import researchpy as rp


# In[6]:


# Assuming you have a DataFrame named 'df' with a 'Club' column

# Sort the DataFrame by the 'Club' column
df = df.sort_values('Club')

# Reset the index to ensure sequential order
df = df.reset_index(drop=True)

# Create a dictionary to store the club IDs
club_ids = {}

# Assign IDs to each club
for index, row in df.iterrows():
    club = row['Club']
    if club not in club_ids:
        club_ids[club] = len(club_ids) + 1
    df.at[index, 'Club ID'] = club_ids[club]

    
df['Club ID'] = df['Club ID'].astype(int)


# In[56]:


df.head()


# In[57]:


rp.codebook(df)

# The `Salary in EU` variable will be our outcome variable of interest (DV).I can look at the weight of the rat pups based on their sex and 
# the treatment group using the .summary_cont method in the `researchpy` package. This will give us a comparison of our 
# N and mean, as well as our measures of variance/spread + confidence interval around the variables.
# In[8]:

rp.summary_cont(df.groupby(["Club","Position_1"])["Salary_Euro"])


# ### Here are few interpretaion below:
# #### Ajax Defenders vs. Barcelona Defenders:
#  
#  Ajax Defenders: The mean salary for Ajax's defenders is approximately 1.33 million.
#  Barcelona Defenders: The mean salary for Barcelona's defenders is approximately 9.98 million.
#  Interpretation: Barcelona's defenders have a significantly higher mean salary compared to Ajax's defenders, indicating that Barcelona may have invested more in their defenders or have higher-valued players in that position.
#  Arsenal Forward vs. Paris Saint-Germain Forward:
#  
#  Arsenal Forward: The mean salary for Arsenal's forwards is approximately 6.18 million.
#  Paris Saint-Germain Forward: The mean salary for Paris Saint-Germain's forwards is approximately 12.73 million.
#  Interpretation: Paris Saint-Germain's forwards have a higher mean salary compared to Arsenal's forwards, suggesting that Paris Saint-Germain may have invested more in their attacking players or have higher-valued forwards.
#  Bayern Munich Goalkeeper vs. Napoli Goalkeeper:
#  
#  Bayern Munich Goalkeeper: The mean salary for Bayern Munich's goalkeepers is approximately 8.73 million.
#  Napoli Goalkeeper: The mean salary for Napoli's goalkeepers is approximately 2.13 million.
#  Interpretation: Bayern Munich's goalkeepers have a higher mean salary compared to Napoli's goalkeepers, indicating that Bayern Munich may have invested more in their goalkeeping position or have higher-valued goalkeepers.
#  Colorado Rapids Defender vs. Porto Defender:
#  
#  Colorado Rapids Defender: The mean salary for Colorado Rapids' defenders is approximately 272,747.
#  Porto Defender: The mean salary for Porto's defenders is approximately 1.18 million.
#  Interpretation: Porto's defenders have a higher mean salary compared to Colorado Rapids' defenders, suggesting that Porto may have more expensive or higher-valued defenders on average.
#  Ajax Midfielder vs. Napoli Midfielder:
#  
# Ajax Midfielder: The mean salary for Ajax's midfielders is approximately 2.05 million.
#  Napoli Midfielder: The mean salary for Napoli's midfielders is approximately 3.10 million.
#  Interpretation: Napoli's midfielders have a higher mean salary compared to Ajax's midfielders, indicating that Napoli may have invested more in their midfield position or have higher-valued midfield players.
#  
#  
# 

# In[58]:


boxplot1 = df.boxplot(["Salary_Euro"], by = ["Club","Club ID"],
                     figsize = (16, 9),
                     showmeans = True,
                     notch = True)
boxplot1.set_xlabel("Club")
boxplot1.set_ylabel("Salary_Euro")
plt.show()


# In[59]:


boxplot1 = df.boxplot(["Salary_Euro"], by = ["Position_1"],
                     figsize = (16, 9),
                     showmeans = True,
                     notch = True)
boxplot1.set_xlabel("Club")
boxplot1.set_ylabel("Salary_Euro")
plt.show()


# In[60]:


from statsmodels.formula.api import ols

# ## Pooled Model
pooled_model = ols("Salary_Euro ~ Age + C(Club) + C(Position_1)", df).fit()
pooled_model.summary()


# The results of the analysis indicate that our model is somewhat effective in explaining the variations in player salaries. I have found some statistically significant relationships between the independent variables (club, position, and age) and the salaries of the players.
#  
# The R-squared value of 0.388 suggests that approximately 38.8% of the differences in player salaries can be explained by the factors included in the model. This means that there are other factors beyond the ones I considered that also influence salary.
#  
# The coefficients provide insights into how different factors affect player salaries. For example, the intercept term represents the estimated salary when all other factors are zero. It shows that the base salary is around -14.96 million Euros, indicating that players start with a negative salary, which doesn't make sense. This could be due to the way the model is set up or missing variables.
#  
# The coefficients for club and position categories tell us how being part of a particular club or playing in a specific position affects player salaries compared to a reference category. For instance, being associated with Barcelona is associated with a salary increase of approximately 8.948 million Euros. However, some of these effects may not be statistically significant, meaning that they could be due to chance rather than a real relationship.
#  
# The age coefficient shows that, on average, each additional year of age is associated with a salary increase of approximately 628,100 Euros. This relationship is statistically significant, indicating that age has a meaningful impact on player salaries.
#  
# There are some limitations to our model. The standard errors, which measure the uncertainty in our coefficient estimates, are quite small. This could suggest that our estimates are overly precise and may not accurately capture the true relationships between the variables.
#  
# In addition, our analysis suggests that the residuals (the differences between actual and predicted salaries) may not follow a normal distribution, meaning that our model may not fully capture all the factors influencing player salaries.
#  
# To further improve our understanding, it would be valuable to consider the intraclass correlation coefficient (ICC) to examine the influence of group-level factors on salaries. This could help us account for variations that are specific to certain clubs or positions.
#  
# In summary, while our model shows some significant relationships between clubs, positions, age, and player salaries, I should interpret these findings cautiously and consider further analysis to address potential limitations and improve the accuracy of our predictions.
# 
# #### Intraclass Correlation Coefficient (ICC)

# In[61]:


import statsmodels.formula.api as smf
formula = 'Salary_Euro ~ 1'
null_model = smf.mixedlm(formula, data=df, groups='Club ID')
null_results = null_model.fit()

null_results.summary()


# In[62]:


# The manual calculation using the numbers from our output is below:
28330787843437.777/(28330787843437.777+71884860372409.5938)


# The output suggests that a random intercept model was fitted to the data. The estimated group-level variance indicates that approximately 28% of the variation in Salary_Euro is due to differences between clubs. The remaining 72% is attributed to individual-level factors. The intercept coefficient is significant, indicating a significant average difference in Salary_Euro across all clubs. Considering the hierarchical structure of the data, a Linear Mixed Effects Model is appropriate to account for both group-level and individual-level effects. This provides a comprehensive understanding of the factors influencing Salary_Euro.

# In[63]:


# Random intercept model using Club ID for our groups. 
# REML is an optional MLE method for estimating variance components in models with random effects when your fixed effects change between models, 
#so you can set it to false if you choose in the fit parameter.
ri_model2 = smf.mixedlm("Salary_Euro ~ Age +Goals+ Matches_Played+ Num_Games_Started+ Minutes_Played+                         Minutes_Played_Divided_By_90+ Goals+ Assists  +Goals_And_Assists+ Non_Penalty_Goals+                         Penalty_Kicks_Made+ Penalty_Kicks_Attempted+ Yellow_Cards+ Red_Cards+ Expected_Goals+                         Non_Penalty_Expected_Goals+ Expected_Assisted_Goals+ Non_Penalty_Expected_Goals_Plus_Assisted_Goals+                         Progressive_Carries+ Progressive_Passes+ Progressive_Passes_Received+ Goals_Against+ Shots_On_Target_Against+                         Goalie_Saves+ Shots_Attempted+ Shots_On_Target+ Average_Shot_Distance_Yards+ Passes_Completed+ Passes_Attempted+                         Total_Passing_Distance_Yards+ Num_of_Players_Tackled+ Tackles_Won+ Blocks+ Shots_Blocked+ Passes_Blocked+                         Touches+ Take_Ons_Attempted+ Take_Ons_Succeded+ Carries+ Progressive_Carrying_Distance_Yards +  C(Club)+ C(Goalie)", df, groups= "Club ID").fit()
ri_model2.summary()


# The intercept coefficient (-10225825.118) represents the baseline salary when all other variables are zero. However, it is not statistically significant at the conventional 0.05 significance level (p-value = 0.102).
# # 
#  * The coefficients for the Club variables (e.g., C(Club)[T.Arsenal], C(Club)[T.Barcelona]) indicate the differences in average salaries compared to a reference club (which is not explicitly mentioned in the output). However, none of these coefficients are statistically significant at the conventional 0.05 significance level, suggesting that the club a player belongs to does not have a significant impact on salary.
#  
#  * The Age coefficient (408420.480) is statistically significant (p-value < 0.001), indicating that older players tend to have higher salaries, all other factors being equal.
#  
#  * Some other statistically significant coefficients include Red_Cards (3715915.207), Shots_Attempted (-386980.485), Shots_On_Target (1007739.948), and Take_Ons_Attempted (248840.792), suggesting that these variables have a significant impact on player salaries.
#  
#  * The random intercept variance (Club ID Var) is extremely large (31320096573274.289), indicating substantial variation in salaries across different clubs.
#   
# The likelihood value (-2762.0806) in the model output measures the goodness of fit, indicating how well the model predicts the observed data. However, the lack of model convergence suggests that the optimization algorithm may not have found the best solution, and further refinements may be needed.
#  
#  The output provides a summary of the model's performance and significance of predictor variables. It includes the number of observations (204) and indicates the presence of both fixed and random effects. The scale represents the variance of the residuals, indicating unexplained variability.
#  
# The coefficient table presents estimates, standard errors, p-values, and confidence intervals for each predictor variable. However, interpreting these results should consider the study context and research question.
#  
# In summary, the output provides insights into the model's fit and predictor significance but requires caution due to convergence issues. Further analysis and refinements are necessary for improved reliability and accurate conclusions.
# 
# #### Random Slope Model

# In[64]:


rs_model = smf.mixedlm("Salary_Euro ~ Goals+ Age +Minutes_Played  + Take_Ons_Succeded+Take_Ons_Attempted+Tackles_Won+ Shots_On_Target+ Shots_Attempted+                        Red_Cards+Yellow_Cards+ C(Club)+ C(Goalie)", df, groups= "Club ID",
                     vc_formula = {"Position_1" : "0 +(Position_1)"}).fit()

rs_model.summary()


# Regarding significance, the p-values associated with each predictor coefficient indicate the statistical significance of the predictor variables in relation to the outcome variable (Salary_Euro). A low p-value suggests a significant impact, while a high p-value suggests a lack of significance. The coefficients with low p-values, such as Age and Shots_On_Target, indicate significant effects on Salary_Euro.
# 
# #### Variance:
# 
# "Club ID Var": The estimated variance associated with the different clubs in relation to Salary_Euro is approximately 4.05 x 10^28 (405,406,207,681,180,625.0). This indicates a significant variability between clubs regarding their impact on Salary_Euro.
# 
# #### Covariance:
# "Club ID x C(Position_1)[T.Forward] Cov": The covariance between the random effects of different clubs and the forward position is approximately 7.37. This suggests a moderate association between certain clubs and Salary_Euro for forward players.
#  
#  "C(Position_1)[T.Forward] x C(Position_1)[T.Goalkeeper] Cov": The covariance between the random effects of forward and goalkeeper positions is approximately 6.76 x 10^12 (6,759,491,057,657.06). This indicates a positive association or overlap in Salary_Euro between these positions.
#  
# "Club ID x C(Position_1)[T.Midfielder] Cov": The covariance between the random effects of different clubs and the midfielder position is approximately 11.02. This suggests a certain level of association between certain clubs and Salary_Euro for midfielders.
#  
# "C(Position_1)[T.Goalkeeper] x C(Position_1)[T.Midfielder] Cov": The covariance between the random effects of goalkeeper and midfielder positions is approximately 9.57 x 10^11 (956,695,740,134.065). This indicates a potential association or interaction between these positions in relation to Salary_Euro.
#  
# These variance components and covariances provide insights into the variability and relationships between different groups (clubs) and positions (forward, goalkeeper, midfielder) in the model. They help quantify the extent of variability and potential associations, contributing to a more comprehensive understanding of the mixed-effects model.
# 
# #### Model comparision
# 
# The log-likelihood values of the three models are as follows:
#  
# * Model 1 (Random Intercept Model): Log-Likelihood = -2761.7907
# * Model 2 (Random Slope Model): Log-Likelihood = -3219.8320
# * Model 3 (Random Slope, Random Intercept Model): Log-Likelihood = -3236.2434
#  
# The log-likelihood is a measure of how well the model fits the data. A higher log-likelihood indicates a better fit, as it represents a higher probability of observing the given data under the model.
#  
# Comparing the log-likelihood values of the three models, we can see that Model 1 (Random Intercept Model) has the highest log-likelihood (-2761.7907), followed by Model 2 (Random Slope Model) with a lower log-likelihood (-3219.8320), and Model 3 (Random Slope, Random Intercept Model) with the lowest log-likelihood (-3236.2434).

#  C(Club)[T.Arsenal]:
#  
#  p-value: 0.065
# The predictor representing the "Arsenal" club has a p-value of 0.065. Since this p-value is greater than the conventional significance level of 0.05, I would consider the effect of being in the Arsenal club on salary to be statistically non-significant. In other words, there is not enough evidence to conclude that being in the Arsenal club has a significant impact on salary.
#  
# C(Club)[T.Barcelona]: 
# p-value: 0.000
# The predictor representing the "Barcelona" club has a p-value of 0.000, which is less than 0.05. Therefore, I can consider the effect of being in the Barcelona club on salary to be statistically significant. Players in Barcelona, on average, are expected to have a significantly higher salary compared to the reference category.
# 
# C(Club)[T.Bayern Munich]:
#  
# p-value: 0.004
# The predictor representing the "Bayern Munich" club has a p-value of 0.004, which is less than 0.05. Hence, the effect of being in Bayern Munich on salary is statistically significant. Players in Bayern Munich, on average, are expected to have a significantly higher salary compared to the reference category.
#  
# C(Club)[T.Colorado Rapids]: 
# p-value: 0.060
# The predictor representing the "Colorado Rapids" club has a p-value of 0.060. Since this p-value is above 0.05, I consider the effect of being in the Colorado Rapids club on salary to be statistically non-significant. There is insufficient evidence to suggest a significant impact on salary for players in the Colorado Rapids.
#  
# C(Club)[T.Napoli]: 
# p-value: 0.754
# The predictor representing the "Napoli" club has a p-value of 0.754, which is greater than 0.05. Thus, the effect of being in the Napoli club on salary is statistically non-significant. There is no strong evidence to indicate a significant impact on salary for players in Napoli.
#  
# C(Club)[T.Paris Saint-Germain]: 
# p-value: 0.000
# The predictor representing the "Paris Saint-Germain" club has a p-value of 0.000, which is less than 0.05. Therefore, the effect of being in Paris Saint-Germain on salary is statistically significant. Players in Paris Saint-Germain, on average, are expected to have a significantly higher salary compared to the reference category.
#  
# C(Club)[T.Porto]: 
# p-value: 0.122
# The predictor representing the "Porto" club has a p-value of 0.122. Since this p-value is greater than 0.05, I consider the effect of being in the Porto club on salary to be statistically non-significant. There is not enough evidence to suggest a significant impact on salary for players in Porto.
#  
# C(Goalie)[T.Yes]:
#  p-value: 0.626
# The predictor representing whether the player is a goalkeeper (Goalie) has a p-value of 0.626, which is greater than 0.05. Therefore, I consider the effect of being a goalkeeper on salary to be statistically non-significant. There is no strong evidence to indicate a significant impact on salary for goalkeepers.
#  
#  Goals, Age, Minutes_Played, Take_Ons_Succeded, Take_Ons_Attempted, Tackles_Won, Shots_On_Target, Shots_Attempted, Red_Cards, Yellow_Cards:
#  
#  All these predictors have associated p-values less than 0.05 (ranging from 0.008 to 0.002), indicating that they have statistically significant effects on salary.
#  
# ### Random Slope, Random Intercept Model
# 

# In[65]:


rs_model = smf.mixedlm("Salary_Euro ~ Age +Minutes_Played  + Take_Ons_Succeded+Take_Ons_Attempted+Tackles_Won+ Shots_On_Target+                        Shots_Attempted+Red_Cards+Yellow_Cards+ C(Club)+ C(Goalie)", df, groups= "Club ID",
                     re_formula = "1+C(Position_1)").fit()
rs_model.summary()


# Regarding significance, the p-values associated with each predictor coefficient indicate the statistical significance of the predictor variables in relation to the outcome variable (Salary_Euro). A low p-value suggests a significant impact, while a high p-value suggests a lack of significance. The coefficients with low p-values, such as Age and Shots_On_Target, indicate significant effects on Salary_Euro.
# 
# #### Variance:
#  
# "Club ID Var": The estimated variance associated with the different clubs in relation to Salary_Euro is approximately 4.05 x 10^28 (405,406,207,681,180,625.0). This indicates a significant variability between clubs regarding their impact on Salary_Euro.
# 
# #### Covariance:
# 
# "Club ID x C(Position_1)[T.Forward] Cov": The covariance between the random effects of different clubs and the forward position is approximately 7.37. This suggests a moderate association between certain clubs and Salary_Euro for forward players.
#  
# "C(Position_1)[T.Forward] x C(Position_1)[T.Goalkeeper] Cov": The covariance between the random effects of forward and goalkeeper positions is approximately 6.76 x 10^12 (6,759,491,057,657.06). This indicates a positive association or overlap in Salary_Euro between these positions.
#  
# "Club ID x C(Position_1)[T.Midfielder] Cov": The covariance between the random effects of different clubs and the midfielder position is approximately 11.02. This suggests a certain level of association between certain clubs and Salary_Euro for midfielders.
#  
# "C(Position_1)[T.Goalkeeper] x C(Position_1)[T.Midfielder] Cov": The covariance between the random effects of goalkeeper and midfielder positions is approximately 9.57 x 10^11 (956,695,740,134.065). This indicates a potential association or interaction between these positions in relation to Salary_Euro.
#  
# These variance components and covariances provide insights into the variability and relationships between different groups (clubs) and positions (forward, goalkeeper, midfielder) in the model. They help quantify the extent of variability and potential associations, contributing to a more comprehensive understanding of the mixed-effects model.
# 
# 
# #### Model comparison
# 
# The log-likelihood values of the three models are as follows:
#  
# * Model 1 (Random Intercept Model): Log-Likelihood = -2761.7907
# * Model 2 (Random Slope Model): Log-Likelihood = -3219.8320
# * Model 3 (Random Slope, Random Intercept Model): Log-Likelihood = -3236.2434
#  
# The log-likelihood is a measure of how well the model fits the data. A higher log-likelihood indicates a better fit, as it represents a higher probability of observing the given data under the model.
#  
# Comparing the log-likelihood values of the three models, I can see that Model 1 (Random Intercept Model) has the highest log-likelihood (-2761.7907), followed by Model 2 (Random Slope Model) with a lower log-likelihood (-3219.8320), and Model 3 (Random Slope, Random Intercept Model) with the lowest log-likelihood (-3236.2434).
