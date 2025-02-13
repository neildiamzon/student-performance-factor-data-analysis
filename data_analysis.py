#!/usr/bin/env python
# coding: utf-8

# # COMP801 - ASSIGNMENT #3 DATA ANALYSIS
# ## Student Performance Factors
# 
# ### Independent Variables (Numerical and Categorical)
# - Hours_Studied	- Number of hours spent studying per week.
# - Attendance - Percentage of classes attended.
# - Parental_Involvement - Level of parental involvement in the student's education (Low, Medium, High).
# - Access_to_Resources - Availability of educational resources (Low, Medium, High).
# - Extracurricular_Activities - Participation in extracurricular activities (Yes, No).
# - Sleep_Hours - Average number of hours of sleep per night.
# - Previous_Scores - Scores from previous exams.
# - Motivation_Level - Student's level of motivation (Low, Medium, High).
# - Internet_Access - Availability of internet access (Yes, No).
# - Tutoring_Sessions - Number of tutoring sessions attended per month.
# - Family_Income - Family income level (Low, Medium, High).
# - Teacher_Quality - Quality of the teachers (Low, Medium, High).
# - School_Type - Type of school attended (Public, Private).
# - Peer_Influence - Influence of peers on academic performance (Positive, Neutral, Negative).
# - Physical_Activity - Average number of hours of physical activity per week.
# - Learning_Disabilities - Presence of learning disabilities (Yes, No).
# - Parental_Education_Level - Highest education level of parents (High School, College, Postgraduate).
# - Distance_from_Home - Distance from home to school (Near, Moderate, Far).
# - Gender - Gender of the student (Male, Female).
# ### Dependent Variable
# Exam_Score	Final exam score.

# ### Import packages

# In[2]:


import pandas as pd


# ### Use Pandas to read csv

# In[3]:


data = pd.read_csv("C:/Users/admin/OneDrive - Auckland Institute of Studies/AIS files/COMP801 - Research Methods/Assignments/Assignment#3 Data Analysis/Datasets/raw dataset/spf.csv")


# In[4]:


data


# ### Describe dataframe to view the Descriptive analysis

# In[5]:


data.describe().T


# In[ ]:




