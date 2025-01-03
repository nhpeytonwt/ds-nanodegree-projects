# Arvato Financial Services: Customer Segmentation and Prediction

## Project Links
- [GitHub Project Repository](https://github.com/nhpeytonwt/ds-nanodegree-projects/tree/main/Project-4-Capstone)
- [Medium Blog Post](https://medium.com/@njhpeyton/can-data-science-help-to-find-customers-hiding-in-plain-sight-62589f61554c)

## Required Libraries
- sys
- numpy
- pandas
- matplotlib
- mpl_toolkits
- seaborn
- sklearn
- pickle
- requests
- tarfile

## Files in repository:
- Arvato Project Workbook.ipynb: Notebook to run project code.
- arvato_functions.py: Functions imported into Arvato Project Workbook.ipynb.
- DIAS Attributes - Values 2017.xlsx: Mapping for data attribute codes. 
- DIAS Information Levels - Attributes 2017.xlsx: Detailed attributes.
- mailout_test_preds.xlsx: Predicted values from final model selected.
- terms_and_conditions/terms.pdf: Arvato terms and conditions.
- terms_and_conditions/terms.md: Arvato terms and conditions.
- README.md: Contains description of project.

## Qualitative Overview: CRISP-DM

### Business Understanding
- Arvato Financial Services, a mail-order sales company in Germany, seeks to better understand their customer base and predict which individuals are most likely to respond to future campaigns.
- This project focuses on two objectives:
  1. **Customer Segmentation**: Analyze how customers differ from the general population using unsupervised learning techniques.
  2. **Response Prediction**: Build a predictive model to determine which targets of a marketing campaign are most likely to become customers.

### Data Understanding
- The data comprises four datasets:
  - **AZDIAS**: Demographics of the general German population.
  - **CUSTOMERS**: Demographics of existing Arvato customers.
  - **MAILOUT_TRAIN**: Targeted marketing campaign data with response labels.
  - **MAILOUT_TEST**: Targeted marketing campaign data without response labels (used for predictions).
- Data is provided in `.csv` format and includes household, building, and neighborhood-level information.

### Data Preparation
The ETL pipeline performs the following tasks:
- Replace encoded "unknown" values as missing (NaN).
- Drop columns with more than 30% missing values.
- Encode categorical variables using one-hot encoding.
- Impute remaining missing data as necessary.
- Standardize numeric variables.

### Modeling
#### Unsupervised Learning (Customer Segmentation)
- **PCA**: Principal Component Analysis reduces the dimensionality of the data to identify latent structures. Key findings suggest diminishing returns to the addition of principal components beyohn 10 or so.
- **K-Means Clustering**: Identifies distinct clusters within the general population and customer data. Results highlight meaningful demographic differences between customers and non-customers.
- In addition, we examined the relationship between PCA and K-Means clusters and find that the general conclusiosn are similar between both approaches. Very nice!

#### Supervised Learning (Response Prediction)
- A **Random Forest Classifier** is used to predict customer responses to marketing campigns:
  - Grid search optimizes hyperparameters on `n_estimators`, `max_depth`, and `min_samples_split`.
  - ROC-AUC score is used to evaluate the model.

### Evaluation
Model results are evaluated based on:
- **Precision**: Proportion of true positive predictions.
- **Recall**: Proportion of true positives correctly identified.
- **ROC-AUC Score**: Overall ability to distinguish between classes.

### Deployment
Predictions for the **MAILOUT_TEST** dataset are saved in `mailout_test_preds.csv`. Key deliverables include:
- A Medium blog post summarizing findings and actionable insights.
- All code and documentation provided in this repository.

## Acknowledgements
- Thanks to **Udacity** and **Bertelsmann Arvato Analytics** for providing the dataset and project framework.
- Additional thanks to the authors of open-source libraries used in this project.