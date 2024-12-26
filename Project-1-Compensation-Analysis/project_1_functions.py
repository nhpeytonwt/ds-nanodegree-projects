# Import libraries
import pandas as pd, numpy as np, os, shutil as sh
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV


# --------------------------------------------------------------------------------------------------------------------------------------------

# Reading excel file into a dictionary of dataframes
def read_excel_file(file_name):
    _ = sh.copy(file_name, 'temp.xlsx')
    xls = pd.ExcelFile('temp.xlsx')
    d = {}
    sheets = xls.sheet_names

    for sheet in sheets:
        d[sheet] = pd.read_excel('temp.xlsx', sheet_name=sheet)
    
    xls.close()
    os.remove('temp.xlsx')  
    return d

# --------------------------------------------------------------------------------------------------------------------------------------------

# For formatting displayed salary values as currencies
def currency_fmt(x, pos):
    return '${:,.0f}'.format(x)

# --------------------------------------------------------------------------------------------------------------------------------------------

# Categorizes salary into bands for salary by country chart
def categorize_salary(salary, bands):
    for band, (lower, upper) in bands.items():
        if lower <= salary < upper:
            return band

# --------------------------------------------------------------------------------------------------------------------------------------------

# Years of experience contains numerical and categorical vals, so we need to convert the categorcal vals
def convert_yrs_exp(value):
    if value == 'Less than 1 year':
        return 0
    elif value == 'More than 50 years':
        return 51
    else:
        return value

# --------------------------------------------------------------------------------------------------------------------------------------------

# Generate dummy variables, either splitting by semicolon for columns with multiple entries, or generating them normally
def gen_dummies(df, column):
    # Split by semicolon if the column contains multiple values
    if df[column].str.contains(';').any():
        split_data = df[column].str.get_dummies(sep=';')
        split_data = split_data.add_prefix(f'{column}_')
        df = pd.concat([df, split_data], axis=1)
        df.drop(column, axis=1, inplace=True)
    # Otherwise generate dummy normally
    else:
        df = pd.get_dummies(df, columns=[column], prefix=[column])
    return df

# --------------------------------------------------------------------------------------------------------------------------------------------

# Function goes column-by-column and imputes missing values using a lasso regression
# Basic idea: Find the columns that all have *complete* values where the target column has missing values, and use those for imputation
def impute_missing_vals(df, target_col,columns_to_exclude, random_seed):
    # For the target column, find which rows have missing (we want to fill these) and which have complete values
    missing_data = df[df[target_col].isnull()]
    complete_data = df.dropna(subset=[target_col])
    
    # Identify *columns with no missing values* in the rows where the *target column has missing values*
    potential_features = missing_data.drop(columns=columns_to_exclude + [target_col]).dropna(axis=1).columns.tolist()
    
    # Get X and y values over the the range where we have complete data for y, as this is what we'll use to *train* the model
    X_complete = complete_data[potential_features]
    y_complete = complete_data[target_col]
    
    # Get optimal model
    kf = KFold(n_splits=20, shuffle=True, random_state=random_seed)
    lasso_cv = LassoCV(cv=kf, random_state=random_seed)
    lasso_cv.fit(X_complete, y_complete)

    # Use X values where y is missing to impute values for y
    X_missing = missing_data[potential_features]
    df.loc[df[target_col].isnull(), target_col] = lasso_cv.predict(X_missing)
    df_imputed = df
    
    # Evaluate the model results
    y_pred = lasso_cv.predict(X_complete)
    r2 = r2_score(y_complete, y_pred)
    
    # Return the imputed dataframe, the number of imputed values, the best alpha, and the R² score
    return df_imputed, missing_data.shape[0], lasso_cv.alpha_, r2

# --------------------------------------------------------------------------------------------------------------------------------------------