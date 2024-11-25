Heart Disease Analysis and Visualization
Overview
This repository contains a comprehensive exploratory data analysis (EDA) and visualization of a heart disease dataset. The goal of this project is to identify patterns, correlations, and insights into the factors contributing to heart disease. The analysis involves data preprocessing, visualization, and distribution analysis for both categorical and numerical features.

Dataset
The dataset used for this analysis includes various features related to heart health, such as cholesterol levels, blood pressure, chest pain types, and other clinical parameters. The target variable is HeartDisease, which indicates whether a person has heart disease (1) or not (0).

Key Features:
Numerical Features: Age, RestingBP, Cholesterol, MaxHR, Oldpeak, etc.
Categorical Features: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope, etc.
Features of the Analysis
Descriptive Statistics:

Summary of numerical and categorical features.
Separate statistics for individuals with and without heart disease.
Data Visualization:

Heatmaps to identify missing values and feature distributions.
Distribution plots for numerical and categorical features.
Count plots and pie charts for categorical data.
Strip plots for exploring relationships between features and heart disease.
Feature Grouping:

Numerical features are grouped into bins (e.g., RestingBP_Group, Cholesterol_Group) for better visualization and analysis.
Target Analysis:

Distribution of the target variable HeartDisease.
Comparison of feature distributions between individuals with and without heart disease.
Tools and Libraries
The following tools and libraries were used in this project:

Python Libraries:
pandas for data manipulation and preprocessing.
numpy for numerical computations.
matplotlib and seaborn for data visualization.
sklearn.preprocessing for encoding categorical variables.
Visualizations
Heatmaps:

Missing value heatmap.
Mean feature values heatmap segmented by heart disease presence.
Distribution Plots:

Numerical and categorical features to understand their distributions.
Pie Charts:

Proportions of features like Sex, ChestPainType, FastingBS, RestingECG, etc.
Strip Plots:

Feature-specific strip plots showing relationships between features and the presence of heart disease.
How to Use
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/heart-disease-analysis.git
Navigate to the project directory:

bash
Copy code
cd heart-disease-analysis
Install required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the analysis: Open the Jupyter Notebook or Python script provided in the repository.

Results
Insights into key factors that influence heart disease.
Visual representation of data distributions and relationships. 
