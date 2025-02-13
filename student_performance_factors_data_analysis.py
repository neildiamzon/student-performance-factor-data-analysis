#!/usr/bin/env python
# coding: utf-8

# # COMP801 - ASSIGNMENT #3 DATA ANALYSIS
# # 20241360 - NEIL ARONN SARNE DIAMZON 
# ## Student Performance Factors
# 
# ✅ Dataset approved on 18th Oct
# 
# ### Objectives
# 
# The objective of this data analysis is to analyze the impact of various factors (Independent variables) on the students' performance. The final exam score (Dependent variable) will be the main metric for this analysis to represent the academic performance of the student. 
# 
# ### Independent Variables (Numerical and Categorical)
# - `Hours_Studied`	- Number of hours spent studying per week.
# - `Attendance` - Percentage of classes attended.
# - `Parental_Involvement` - Level of parental involvement in the student's education (Low, Medium, High).
# - `Access_to_Resources` - Availability of educational resources (Low, Medium, High).
# - `Extracurricular_Activities` - Participation in extracurricular activities (Yes, No).
# - `Sleep_Hours` - Average number of hours of sleep per night.
# - `Previous_Scores` - Scores from previous exams.
# - `Motivation_Level` - Student's level of motivation (Low, Medium, High).
# - `Internet_Access` - Availability of internet access (Yes, No).
# - `Tutoring_Sessions` - Number of tutoring sessions attended per month.
# - `Family_Income` - Family income level (Low, Medium, High).
# - `Teacher_Quality` - Quality of the teachers (Low, Medium, High).
# - `School_Type` - Type of school attended (Public, Private).
# - `Peer_Influence` - Influence of peers on academic performance (Positive, Neutral, Negative).
# - `Physical_Activity` - Average number of hours of physical activity per week.
# - `Learning_Disabilities` - Presence of learning disabilities (Yes, No).
# - `Parental_Education_Level` - Highest education level of parents (High School, College, Postgraduate).
# - `Distance_from_Home` - Distance from home to school (Near, Moderate, Far).
# - `Gender` - Gender of the student (Male, Female).
# ### Dependent Variable
# - `Exam_Score` - Final exam score.

# ### Import packages

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# ### Use Pandas to read csv

# In[2]:


raw_df = pd.read_csv("spf.csv")


# In[3]:


print("Raw data row count: " + str(len(raw_df)))


# In[4]:


raw_df.head()


# # Data cleaning process
# 

# **Dataset contains null values. We need to check if the null percentage is greater than 80% before dropping the columns/rows. Alternatively, we can drop the rows since removed rows are negligible.**

# In[5]:


null_columns = raw_df.isnull().sum()
null_values_df = pd.DataFrame(null_columns, columns=['Number of null values'])
null_values_df["percentage"] = ((null_values_df[null_values_df["Number of null values"] > 0]/ len(raw_df))*100).round(2)
null_values_df[null_values_df["Number of null values"] > 0]


# **Dropping rows with null values**

# In[6]:


cleaned_df = raw_df.dropna()
print("Data count after dropping null rows: " + str(len(cleaned_df)))


# **Checking for Duplicates and removing them to avoid Data Bias**

# In[7]:


cleaned_df.drop_duplicates()


# **Changing all texts to lowercase to ensure consistency in the dataset**

# In[8]:


# if a column is a string data type, convert to lowercase
cleaned_df = raw_df.map(lambda x: x.lower() if isinstance(x, str) else x)
cleaned_df.head()


# **Segregating numerical and categorical columns**
# We only needed numerical columns

# In[9]:


numerical_columns = cleaned_df.select_dtypes(exclude="object").columns
categorical_columns = cleaned_df.select_dtypes(include="object").columns

print("There are " + str(len(numerical_columns)) + " numerical columns ")

numerical_df = cleaned_df[numerical_columns]
categorical_df = cleaned_df[categorical_columns]


# ## Checking for outliers

# In[10]:


# Provides a summary statistic of the numeric variables to check outliers
summary_df = numerical_df.describe().T
summary_df['median'] = numerical_df.median()
summary_df['mode'] = numerical_df.mode().iloc[0]
summary_df


# ## Using M

# ### Histograms
# **As seen below in the following histograms that there are no outliers or extreme values found within x-axis of each of the numerical columns of the dataset**

# In[11]:


numerical_df.hist(bins=25, figsize=(9, 8), edgecolor='black')
plt.show()


# ### Box Plot
# **As seen below that there are no outliers or extreme values that is way above the median level of each columns**
# The outliers seen in hours_studied are acceptable because there is no anomalies with the numbers as hours_studied are per week and it is still in the acceptable range

# In[12]:


plt.figure(figsize=(14, 5))
numerical_df.boxplot()
plt.show()


# ## Visualizing the Demographics of the dataset

# In[13]:


#gender_chart = categorical.groupby('Gender').size()
#gender_chart.plot.pie( autopct='%1.1f%%', figsize=(4, 5), startangle=140, title='Distribution by Gender')

#gender_chart

fig, ax = plt.subplots(3, 3, figsize=(9,8))

# Pie Chart

gender_data = categorical_df['Gender'].value_counts()
ax[0,0].pie(gender_data, labels=gender_data.index,  autopct='%1.1f%%', startangle=90, colors=['skyblue', 'pink']) 
ax[0,0].grid(True)
ax[0,0].set_title("Gender Distribution")

extracurricular_activities_data = categorical_df['Extracurricular_Activities'].value_counts()
ax[2,0].pie(extracurricular_activities_data, labels=extracurricular_activities_data.index,  autopct='%1.1f%%', startangle=90, colors=['green', 'red']) 
ax[2,0].grid(True)
ax[2,0].set_title("Extracurricular Activities")

learning_disabilities_data = categorical_df['Learning_Disabilities'].value_counts()
ax[2,1].pie(learning_disabilities_data, labels=learning_disabilities_data.index,  autopct='%1.1f%%', startangle=90, colors=['red', 'green']) 
ax[2,1].grid(True)
ax[2,1].set_title("Learning Disabilities")

parental_education_level_data = categorical_df['Parental_Education_Level'].value_counts()
ax[0,2].pie(parental_education_level_data, labels=parental_education_level_data.index,  autopct='%1.1f%%', startangle=90) 
ax[0,2].grid(True)
ax[0,2].set_title("Parental Education Level")

# Bar Charts
order = ['low','medium','high']
parental_involvement_data = categorical_df['Parental_Involvement'].value_counts().reindex(order)
ax[0,1].bar(parental_involvement_data.index, parental_involvement_data, color=['Red', 'Orange', 'Green'])
ax[0,1].grid(True)
ax[0,1].set_title("Parental Involvement")

access_to_resources_data = categorical_df['Access_to_Resources'].value_counts().reindex(order)
ax[1,0].bar(access_to_resources_data.index, access_to_resources_data, color=['Red', 'Orange', 'Green'])
ax[1,0].grid(True)
ax[1,0].set_title("Access to Resources")

family_income_data = categorical_df['Family_Income'].value_counts().reindex(order)
ax[1,1].bar(family_income_data.index, family_income_data, color=['Red', 'Orange', 'Green'])
ax[1,1].grid(True)
ax[1,1].set_title("Family Income")

motivation_level_data = categorical_df['Motivation_Level'].value_counts().reindex(order)
ax[1,2].bar(motivation_level_data.index, motivation_level_data, color=['Red', 'Orange', 'Green'])
ax[1,2].grid(True)
ax[1,2].set_title("Motivation Level")

distance_from_home_data = categorical_df['Distance_from_Home'].value_counts().reindex(['near', 'moderate', 'far'])
ax[2,2].bar(distance_from_home_data.index, distance_from_home_data, color=['Green', 'Red', 'Orange'])
ax[2,2].grid(True)
ax[2,2].set_title("Distance from Home")
# 	Hours_Studied	Attendance	Parental_Involvement	Access_to_Resources	Extracurricular_Activities	Sleep_Hours	
# Previous_Scores	Motivation_Level	Internet_Access	Tutoring_Sessions	Family_Income	Teacher_Quality	School_Type	Peer_Influence	
# Physical_Activity	Learning_Disabilities	Parental_Education_Level	Distance_from_Home	Gender	Exam_Score

fig.tight_layout()


# In[14]:


fig, ax = plt.subplots(2, 3, figsize=(10,9))

ax[0,0].scatter(numerical_df["Hours_Studied"], numerical_df["Exam_Score"], s=5)
ax[0,0].grid(True)
ax[0,0].set_title("Hours_Studied vs Exam Score")

ax[0,1].scatter(numerical_df["Attendance"], numerical_df["Exam_Score"], s=5)
ax[0,1].grid(True)
ax[0,1].set_title("Attendance vs Exam Score")

ax[0,2].scatter(numerical_df["Sleep_Hours"], numerical_df["Exam_Score"], s=5)
ax[0,2].grid(True)
ax[0,2].set_title("Sleep_Hours vs Exam Score")

ax[1,0].scatter(numerical_df["Previous_Scores"], numerical_df["Exam_Score"], s=5)
ax[1,0].grid(True)
ax[1,0].set_title("Previous_Score vs Exam Score")

ax[1,1].scatter(numerical_df["Tutoring_Sessions"], numerical_df["Exam_Score"], s=5)
ax[1,1].grid(True)
ax[1,1].set_title("Tutoring_Sessions vs Exam Score")

ax[1,2].scatter(numerical_df["Physical_Activity"], numerical_df["Exam_Score"], s=5)
ax[1,2].grid(True)
ax[1,2].set_title("Physical Activity hours vs Exam Score")

fig.tight_layout()


# ## Data Analysis - Regression

# ### Correlation Matrix

# In[15]:


correlation_matrix = numerical_df.corr()
print(correlation_matrix)


# ### Plotting the Correlation Matrix for visualization

# In[16]:


plt.figure(figsize=(8, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, linecolor='black')

# Add title for clarity
plt.title('Correlation Matrix')

# Show the plot
plt.show()


# In[17]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import math


# In[18]:


IV_array = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
for index in range(len(IV_array)):
    x = numerical_df[[IV_array[index]]]
    y = numerical_df['Exam_Score']
    model = LinearRegression()
    model.fit(x,y)
    
    numerical_df = numerical_df.copy()
    # Predictions
    numerical_df['Predicted'] = model.predict(x)
    numerical_df['Residuals'] = y - numerical_df['Predicted']
    
    # Number of observations and predictors
    n = len(y)
    p = x.shape[1]
    
    intercept = model.intercept_
    coefficients = model.coef_
    mse = mean_squared_error(y, numerical_df['Predicted'])
    r2 = r2_score(y, numerical_df['Predicted'])
    adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
    
    # F-statistic and p-value
    ssr = sum((numerical_df['Predicted'] - y.mean()) ** 2)  # Sum of Squares for Regression
    sse = sum(numerical_df['Residuals'] ** 2)  # Sum of Squares for Errors
    msr = ssr / p  # Mean Square for Regression
    mse_error = sse / (n - p - 1)  # Mean Square for Errors
    f_statistic = msr / mse_error
    p_value_f = 1 - stats.f.cdf(f_statistic, p, n - p - 1)
    
    # Output results
    print(f"Regression Statistics for: {IV_array[index]} vs Exam Scores")
    print(f"Intercept: {intercept}")
    print(f"Coefficients: {coefficients}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Multiple R: {math.sqrt(r2)}")
    print(f"R-squared: {r2}")
    print(f"Adjusted R-squared: {adjusted_r2}")
    print(f"F-statistic: {f_statistic}")
    print(f"P-value for F-statistic: {p_value_f}")
    print('-------------------------------------------')


# ### Interpretation:
# 
# 1. `Hours_Studied` vs `Exam_Scores` - when hours studied is zero, the predicted exam score is approximately 61.46 *(intercept)*. For every additional hour of studying, the exam scores is preditect to increase by about 0.29 points *(coefficient)*.
#     * Around 19.8% of the variance in the exam scores are explained by the number of hours studied, which could indicate a weak relationship *(r-squared)*. The result is similar to adjusted R-squared.
#     * With an F-statistic of 1635.08 and p-value of  1.1102230246251565e-1 *(almost zero)*, it suggests that the model is statistically significant, but the weak R-squared suggests that the hours studied alone may not be a strong predictor.
# 
#   
# 2. `Attendance` vs `Exam_Score` - when the attendance is zero, the predicted exam score is approximately 51.58 *(intercept)*. For every additional attendance, the exam scores increase by around 0.20 points *(coefficient)*.
#     * Around 33.8% of the variance in the exam scores are explained by the attendance, which could indicate a moderate relationship *(r-squared)*.
#     * With an F-statistic of 3366.99 and p-value of 1.1102230246251565e-16 *(almost zero)*, it suggests that the model is highly significant, and the attendance variable may be a strong predictor.
# 
# 
# 3. `Sleep_Hours` vs `Exam Scores` - when there's no sleep hours, the predicted exam score is 67.55. *(intercept)*. A *coefficient* of -0.045 indicates that a small, negative relationship exists between sleep hours and exam scores.
#     * The sleep hours explain less than 0.03% of the variance in exam scores, suggesting no meaningful relationship *(r-squared)*.
#     * With an F-statistic of 1.91 and p-value of 0.166), the model is not statistically significant *(because p-values >= 0.05)*. Sleep hours do not predict exam scores effectively.
# 
#   
# 4. `Previous_Scores` vs `Exam Scores` - when previous exam score is 0, the predicted exam scores for the current tests ia around 63.68 *(intercept)*. An additional point increase in the previous exam corresponds to about 0.047 increase in exam scores *(coefficient)*.
#     * The previous scores explain only 3.1% of the variance which suggests a weak relationship the exam scores *(r-squared)*.
#     * With an F-statistic of 208.86 and the p-value of 1.1102230246251565e-16 *(almost zero)*, the model is statistically significant, but its practical usefulness is limited.
# 
#   
# 5. `Tutoring_Sessions` vs `Exam Scores` - if tutoring sessions are none, the predicted exam score is about 66.50 *(intercept)*. Each additional tutoring session increases exam scores by about 0.50 points *(coefficient)*.
#     * The tutoring sessions explain 2.4% of the variance, indicating a very weak relationship *(r-squared)*.
#     * With an F-statistic of 165.89 and the p-value of 1.1102230246251565e-16 *(almost zero)*, the model is statistically significant but lacks practical impact *(because the F-statistic is low).*
#   
# 6. `Physical_Activity` vs `Exam Scores` - when physical activity is none, the predicted exam score is about 66.92 *(intercept)*. For each session of a physical activity, exam scores increase by 0.105 points *(coefficient)*.
#     * Since r-squared is 0.00077, it means that physical activity explains less than 0.1% of the variance in exam scores, suggesting almost no relationship.
#     * With an F-statistic of 5.12 and the p-value of  around 0.24, the model is statistically significant, but since F-statistic is very low, the impact or explanatory power of physical activity to exam scores is negligible.

# ## ANOVA 

# In[19]:


from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm


# **ANOVA tables**

# In[20]:


for index in range(len(IV_array)):
    iv = IV_array[index]
    dv = 'Exam_Score'
    formula = f'{dv} ~ C({iv})'
    model = ols(formula, data=numerical_df).fit()
    anova_table = anova_lm(model, typ=2)
    display(anova_table)

