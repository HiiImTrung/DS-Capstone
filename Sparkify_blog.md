
## Predicting Churn for Sparkify Users with Apache Spark

## I. Introduction

In today's competitive music streaming industry, understanding why users churn is essential for retaining customers. Churn, or when users stop using a service, can drastically impact revenue. By identifying at-risk users, companies can take proactive measures to keep them engaged, whether through targeted offers, promotions, or personalized recommendations.

For this project, I used a sample dataset from Sparkify, a fictional digital music service, to predict churn with Apache Spark. This blog post covers the entire process, from data loading and exploration to feature engineering, model training, and evaluation. Let’s dive into the steps I took to build a data pipeline and machine learning model to predict churn, optimize it, and measure its effectiveness.

## II. Project Overview

This project follows the classic data science workflow:

- Problem Definition: Define churn and how we’ll detect it.

- Data Exploration: Understand and explore the dataset.

- Feature Engineering: Create relevant features that capture user behavior.

- Modeling: Train machine learning models to predict churn.

- Evaluation: Evaluate model performance and discuss potential improvements.

- Tool of Choice: Given the dataset's large size, I used Apache Spark to process and analyze data efficiently. Spark's distributed nature allows it to handle data at scale, which is ideal for this project.

## 1. Problem Definition

The Goal

The main objective is to predict which users are likely to churn, meaning they downgrade their service or cancel it altogether. Early detection can allow Sparkify to take action and reduce churn rates.

Defining Churn

In this project, I defined churn based on specific user actions:

- Cancellation Confirmation events in the user log indicate churn.
- Additionally, Downgrade events indicate potential churn as users shift to lower subscription tiers.
- The target variable (churn) is binary:

1 for users who churned,

0 for users who remained.

## 2. Data Exploration

Loading and Cleaning the Data

The data is loaded from a JSON file (mini_sparkify_event_data.json) containing event logs from Sparkify users. This dataset includes information such as:

- userId: Unique identifier for each user
- sessionId: Unique identifier for each session
- page: The type of user interaction (e.g., NextSong, Thumbs Up, Logout)
- song: The song name
- artist: The artist name
- ts: Timestamp of the event
- length: Length of the song played
- level: User subscription level (free or paid)
- gender: User's gender
- location: User's location

These attributes are essential for understanding user behavior and building predictive models for churn.
Since some records have missing values (e.g., userId or sessionId), the first step was to filter out incomplete records to ensure data consistency.


## 3. Feature Engineering

After defining churn, the next step was to create meaningful features. Feature engineering transforms raw data into useful attributes for a machine learning model. Some important features created include:

You can visualize how many songs users play before churning or staying.
- Session Duration: User Activity of the Day by Hour
- Popular songs: Top 10 Popular Songs from the Data
- Thumbs Up / Thumbs Down Count: Captures user satisfaction.
- Demographics: Distribution of user gender and location.
- Churn Indicators: High correlation between certain activities (e.g., "Cancellation Confirmation") and churn.


Creating these features required aggregating data at the user level and handling time-based data. Apache Spark allowed me to efficiently calculate these aggregations even with a large dataset.

Visualizations such as histograms, bar plots, and heatmaps help uncover these insights and guide feature engineering.

Example of some visualizazions:

1. Session Duration
![image.png](attachment:image.png)


2. Distribution of Thumbs up and Thumbs down
![image.png](attachment:image.png)

3. User Gender
![image.png](attachment:image.png)

4. Top 10 Popular Song
![image.png](attachment:image.png)

## 4. Modeling
With the features ready, I moved on to training machine learning models. I experimented with several algorithms available in Spark’s MLlib, including:

- Logistic Regression: This model assumes a linear relationship between the features and the target variable. While it provides a baseline, it may not capture the non-linear patterns in user behavior effectively, leading to lower performance compared to more complex models.

- Decision Tree Classifier: This model can capture non-linear relationships by splitting the data based on feature values. It inherently performs feature selection, which helps in identifying the most informative features. The ability to handle non-linearity and perform feature selection contributes to its superior performance.

- Random Forest Classifier: This ensemble method improves robustness by combining multiple decision trees. However, in this case, the Random Forest did not significantly outperform the single Decision Tree, possibly due to the complexity and overfitting of the trees. Hyperparameter tuning was critical, but the Decision Tree's simplicity proved advantageous.

Since the dataset is large and churn is relatively rare, I used the F1 score as the primary metric to evaluate model performance. The F1 score balances precision and recall, which is critical in cases of imbalanced classes like churn prediction.


## III. Results
After evaluating all models, the Decision Tree Classifier  emerged as the best-performing model with an F1 score of 0.80 on the test set. Here’s a summary of the key results:

![image.png](attachment:image.png)
## IV. Key Findings
The analysis revealed some important behaviors associated with churn:

- Low Engagement: Users who play fewer songs per session or interact with the platform infrequently are more likely to churn.
- Negative Feedback: Users with a high ratio of "Thumbs Down" interactions are at higher risk of churning.
- Frequent Downgrades: Downgrading from paid to free tiers often signals potential churn.

## V. Conclusion and Future Work

## 1. Summary

Based on the evaluation results, the Decision Tree Classifier outperformed both Logistic Regression and Random Forests in predicting churn, achieving the highest F1 score of 0.8049. The reasons behind this superior performance can be summarized as follows:

- Model Complexity and Flexibility Decision Trees are well-suited for capturing non-linear relationships in data. While Logistic Regression assumes linear relationships, which may not always hold true in real-world datasets, Random Forests might not outperform a single decision tree in simpler cases due to the averaging of multiple trees that may smooth out important signals. Decision trees are more flexible in terms of the complexity they can model without assuming linearity.
- Interpretability Decision Trees are known for their interpretability. Each decision rule (split) in a tree is based on a single feature and a threshold, making it easy to trace the logic behind the model's decisions. This transparency is valuable, especially when the goal is to derive actionable business insights (e.g., understanding what user behaviors lead to churn). In contrast, Logistic Regression might be less intuitive when dealing with multiple features, and Random Forests, being an ensemble method, can become complex and harder to interpret due to the combination of many trees.
- Feature Selection Decision Trees inherently perform feature selection at each split, focusing on the most important features that best separate the classes. This feature selection process can be very effective, particularly in datasets with many features, as the tree splits on the most informative ones. Although Random Forests also offer feature importance scores and Logistic Regression provides coefficients, the Decision Tree classifier’s built-in feature selection often leads to better performance when the most predictive features are identified.
- Hyperparameter Tuning The performance of the Decision Tree Classifier was likely enhanced through effective hyperparameter tuning. Parameters such as tree depth, minimum samples per leaf, and the maximum number of features considered at each split can significantly impact model performance. By fine-tuning these parameters, the model was able to better capture the nuances of the data. While Random Forests benefit from ensemble methods, individual decision trees can be tuned more precisely for this particular dataset.
- Model Simplicity and Efficiency Decision Trees are computationally efficient and can handle large datasets quickly, which can make them attractive for real-time predictions in large-scale applications. While Random Forests generally offer better performance by combining many trees, they can be slower to train and harder to deploy at scale, especially in resource-constrained environments. Final Summary In conclusion, the Decision Tree Classifier achieved the highest F1 score and outperformed Logistic Regression and Random Forests in this churn prediction task. Its ability to capture non-linear relationships, its ease of interpretability, the built-in feature selection mechanism, and effective hyperparameter tuning contributed to its superior performance. While Random Forests are generally robust, the simpler and more interpretable Decision Tree model provided the best results for this dataset, making it the optimal choice for this particular churn prediction problem.

## 2. Next Steps

To further improve the model, some future steps could include:

- Handling Imbalance: Using techniques like oversampling, undersampling, or synthetic data generation (SMOTE) to improve the model’s sensitivity to churners.
- Using More Features: Exploring more granular features, such as session-specific behavior or time-based trends.
- Deploying the Model: Integrate the model into a production environment to make real-time predictions and inform Sparkify’s retention strategies.

## VII. Final Thoughts

This project demonstrated the power of Apache Spark for large-scale data processing and machine learning. By efficiently processing large volumes of data, Spark enabled the exploration, transformation, and modeling of complex user behavior patterns at scale.

Building a churn prediction model is just the first step. The insights from this model can help Sparkify take proactive actions to retain users, improve engagement, and ultimately increase revenue. If you're looking to tackle a similar problem, consider using Spark for scalable, efficient processing of large datasets.
