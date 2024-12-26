# Import libraries
import pandas as pd, numpy as np, os, shutil as sh
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoCV


# --------------------------------------------------------------------------------------------------------------------------------------------

def read_excel_file(file_name):
    """ Reads Excel file into a dictionary of dataframes organized by the file's sheet names

    Args:
        file_name: Path of the Excel file

    Returns:
        d: Dictionary of dataframes, where the keys are the file's sheet names
    """
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

def currency_fmt(x, pos):
    """ Formats input values as currencies

    Args:
        x: Value to format
        pos: Tick position of x

    Returns:
        Re-formatted value of x
    """
    return '${:,.0f}'.format(x)

# --------------------------------------------------------------------------------------------------------------------------------------------

def categorize_salary(salary, bands):
    """ Categorizes salary into bands for salary by country chart

    Args:
        salary: Salary to categorize
        bands: Dictionary of salary bands

    Returns:
        Band that the salary falls into
    """
    for band, (lower, upper) in bands.items():
        if lower <= salary < upper:
            return band

# --------------------------------------------------------------------------------------------------------------------------------------------

def convert_yrs_exp(value):
    """ Converts years of experience from categorical to numerical values if applicable

    Args:
        value: Input value, which is either 'Less than 1 year', 'More than 50 years', or a numerical value

    Returns:
        Years of experience, changed to a numerical if input was categorical
    """
    if value == 'Less than 1 year':
        return 0
    elif value == 'More than 50 years':
        return 51
    else:
        return value

# --------------------------------------------------------------------------------------------------------------------------------------------

def gen_dummies(df, column):
    """ Generate dummy variables, either splitting by semicolon for columns with multiple entries, or generating them normally for entires with single values

    Args:
        df: Dataframe that target column is derived from
        column: Column to generate dummies for

    Returns:
        Updated dataframe with dummy variables generated
    """
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

def impute_missing_vals(df, target_col,columns_to_exclude, random_seed):
    """ Function goes column-by-column and imputes missing values using a lasso regression
        Basic idea: Find the columns that all have *complete* values where the target column has missing values, and use those for imputation

    Args:
        df: Dataframe that contains the target column
        target_col: Target column to impute missing values
        columns_to_exclude: Columns ignored in imputation
        random_seed: Ensures reproducibility

    Returns:
        df_imputed: Updated df with missing values imputed
        missing_data.shape[0]: Count of imputed values
        lasso_cv.alpha_: Optimal alpha value from lasso crossval
        r2: R² score of optimal model
    """
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