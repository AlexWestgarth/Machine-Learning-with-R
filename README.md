# Machine-Learning-with-R
An assignment on text based machine learning with R

# Purpose
To demonstrate data analytical skills in R, namely:
 - Loading data
 - Data inspection
 - Model selection
 - Text Data manipulation
 - Text Data analysis
 - Text Data modelling
 
 Additionally the use of business problem statement understanding and transformation into a data-driven report.
 
 # Introduction
 A company wished to perform an analysis into which news reports were fake or real.
 A full data analysis with machine learning was performed via R using R Studio.
 The dataset provided contained approximately 11500 articles, each containing text, tags, author, data and label (real or fake).
 
 Investigation into datasets of similar size and structure reveals that this dataset was closely based on similar sets existing on Kaggle.com
 Use of these additional datasets was excluded from the assignment explicitly by the University, however addition of this data to the analysis would have been a simple task.
 
 Information and techniques used in these similar sets was employed in this analysis with careful consideration of the similarities and differences.
 
 # Method
 The data was investigated in a text editor to inspect any possible data loading errors.
 
 The data was then loaded into a DataFrame object and inspected.
 The first investigation is into the "Label" data, which indicates the article's status of real or fake.
 
 Unlabelled data is then removed, due to the inability to ultilise that data without a proper label.
 
 The imbalance between the two labels is taken note of, as it will require adjustment later in the process to ensure model validity.
 
 Primary accessment of the data assumes that the only usable data to determine label is the "text" data, but to ensure robust analysis, investigation into the other data columns is performed.
 
 Tags and author both imply the possibility of deriving label, but upon closer inspection reveal no observable corrolation between them and the target variable.
 
 In light of this, these columns are dropped from the working dataset to reduce process time.
 
 The data is then formated and prepared such that a model can perform analysis. This is via the Document Term Matrix object and the lemmatization process.
 
 The data is then split into three sets, the original unbalanced set, an upsampled and a downsampled set. A note is made regarding two alternatives for rebalance data, including reasoning for exclusion in this data.
 
 Each of the three prepared datasets are then modelled against three different modelling types; Naive Bayes, Random Forest and Logistic Regression.
 
 The best model and the best balancing method are both selected from the 9 produced models and then used to complete the predicition against the testing data.
