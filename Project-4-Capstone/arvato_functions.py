# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# --------------------------------------------------------

def read_chunks(filepath,chunksize,sep,**kwargs):
    """
    Reads in a CSV file by chunks
    
    Args:
    - filepath (str): Path to CSV.
    - chunksize (int): Rows per chunk.
    - sep (str): Delimiter.
    - **kwargs: Add'l arguemtns for 'pd.read_csv'
    
    Returns:
    -Populted dataframe
    """
    
    # Create list of chunks
    chunk_list = []
    for chunk in pd.read_csv(filepath,chunksize=chunksize,sep=sep,**kwargs):
        chunk_list.append(chunk)
    
    # Combine into dataframe
    df = pd.concat(chunk_list,ignore_index=True)
    
    return df
    
# --------------------------------------------------------
    
def noncommon_cols(dfs):
    """
    Find the unique columns amongst all dataframes
    
    Args:
    - dfs (dict): Dictionary of dataframes.
    """
    # Pull columns for each dataframe
    cols_dict = {name: set(df.columns) for name, df in dfs.items()}

    # Find the common columns between them
    common_cols = set.intersection(*cols_dict.values())

    # Use each dataframe's columns and the common columns between them to find the non-common columns
    for name, df_cols in cols_dict.items():
        noncommon = df_cols - common_cols
        if len(noncommon) == 0:
            print(f"{name} has 0 non-common columns.\n")
        else:
            print(f"{name} has {len(noncommon)} non-common columns:\n{noncommon}\n")
            
# --------------------------------------------------------

def gen_unknown_map(filepath='DIAS Attributes - Values 2017.xlsx'):
    """
    Reads DIAS Attributes - Values 2017.xlsx file, identifies NaN mapping by parsing the 'Meaning' column, and returns dictionary.
    
    Args:
    - atts_path (str): Path to the mapping file.
    
    Returns:
    - unknown_map (dict): A dictionary mapping each attribute to the list of codes that represent 'unknown' or 'no classification'.
    """
    
    df = pd.read_excel(filepath,header=[1])  

    # Forward-fill attribute name
    df['Attribute'] = df['Attribute'].fillna(method='ffill')

    # Filter rows that mention "unknown" or "no " in 'Meaning'
    df['Meaning'] = df['Meaning'].fillna('').astype(str)
    mask_unknown = df['Meaning'].str.lower().str.contains('unknown|no ')
    df_unknown = df[mask_unknown].copy()
    
    # Build dictionary for each attribute, aggregating unknown codes
    unknown_map = {}
    for attribute in df_unknown['Attribute'].unique():
        # Attribute maps to the columns in dfs, so we'll loop through them and derive the values that reflect missing observations
        rows = df_unknown.loc[df_unknown['Attribute'] == attribute, 'Value']
        
        codes = []
        for raw_val in rows:
            # Some rows might have multiple codes like "-1, 0"
            for part in str(raw_val).split(','):
                part = part.strip()
                
                # Try converting to integer, else keep string
                try:
                    part = int(part)
                except ValueError:
                    pass
                codes.append(part)
        
        # Generate unique set of codes and sort
        unique_codes = sorted(set(codes), key=lambda x: str(x))
        unknown_map[attribute] = unique_codes

    return unknown_map

# --------------------------------------------------------

def missing_cols(dfs, threshold=20, remove=False):
    """
    Find and drop columns with missing data if they exceed the threshold in *any* of the dataframes in dfs

    Args:
    - dfs (dict): Dictionary of dataframes.
    - threshold (float): Missing-value threshold in pct.
    - remove (bool): If True, drops flagged cols in each dataframe.
    
    Retruns:
    - dfs (dict): Updated dict of dataframes (if remove flag was set to true)
    """
    
    # Identify columns above threshold in each df
    cols = set()
    for name, df in dfs.items():
        percent_missing = df.isnull().mean() * 100
        flagged = percent_missing[percent_missing >= threshold].index
        cols.update(flagged)
    
    # Print flagged columns
    if len(cols) == 0:
        print(f"No columns found with >= {threshold}% missing data.\n")
    else:
        print(f"Columns with >= {threshold}% missing values (in at least one DataFrame):")
        print(sorted(cols), "\n")
    
    # If remove=True, drop these columns in each DataFrame
    if remove and len(cols) > 0:
        for name, df in dfs.items():
            before = df.shape[1]
            dfs[name] = df.drop(columns=cols.intersection(df.columns), axis=1)
            after = dfs[name].shape[1]
            print(f"Dropped {before - after} columns from '{name}'. Now {after} columns remain.")
        print()

    # Return the updated dictionary
    return dfs

# --------------------------------------------------------

def encode_cats(df,name):
    """
    Converts all object dtype columns in each DataFrame to dummy vars.

    Args:
    - df (dataframe): Input dataframe.
    
    Returns:
    - df (dataframe): Output dataframe with encoded categorical columns.
    """
    import pandas as pd
        
    cat_cols = df.select_dtypes(include=['object']).columns

    if len(cat_cols) == 0:
        print(f'{name}: No categorical variables found.')
    else:
    # Create dummy vars
        dummies = pd.get_dummies(df[cat_cols], drop_first=True)
        df = pd.concat([df.drop(columns=cat_cols), dummies], axis=1)

        print(f"Encoded {len(cat_cols)} categorical cols ({len(dummies.columns)} new cols) in '{name}'.")

        return df

# --------------------------------------------------------

def std_float(dfs):
    """
    Standardize the float64 columns in 'mailout_train' (ignoring 'RESPONSE') and apply scaler to other dataframes.

    Args:
        dfs (dict): Dictionary of dataframes.
                    
    Returns:
        dfs (dict): Updated dictionary of dataframes with scaled values.
    """
    # Identify float64 columns in mailout_train
    train_df = dfs['mailout_train']
    
    # Find columns to scale
    features_to_scale = [
        col for col in train_df.columns
        if train_df[col].dtype == np.float64 and col != 'RESPONSE'
    ]
    
    if not features_to_scale:
        print("No float64 columns found in 'mailout_train' to scale.")
        return dfs  
    
    # Transform mailout_train
    scaler = StandardScaler()
    scaler.fit(train_df[features_to_scale])
    dfs['mailout_train'][features_to_scale] = scaler.transform(train_df[features_to_scale])
    print(f"Scaled {len(features_to_scale)} float64 columns in 'mailout_train'.")
    
    # Transform other dataframes
    for key in ['azdias', 'customers', 'mailout_test']:
        dfs[key][features_to_scale] = scaler.transform(dfs[key][features_to_scale])
        print(f"Scaled {len(features_to_scale)} float64 columns in '{key}'.")
    
    return dfs

# --------------------------------------------------------

def median_impute(dfs, ignore_cols=None):
    """
    Impute missing values in each DataFrame with the median of that column.
    
    Args:
        dfs (dict): Dictionary of dataframes.
        ignore_cols (list): List of columns to skip imputation (e.g., ['RESPONSE']).
        
    Returns:
        dfs (dict): Updated dictionary of dataframes.
    """
    
    imputer = SimpleImputer(strategy='median')
    
    if ignore_cols is None:
        ignore_cols = []
    for name in dfs.keys():
        # Ientify columns to impute
        cols_impute = [col for col in dfs[name].columns if col not in ignore_cols]
        
        if not cols_impute:
            print(f"No columns to impute in '{name}' after ignoring {ignore_cols}. Skipping.")
            continue
        
        # Perform imputation
        dfs[name][cols_impute] = imputer.fit_transform(dfs[name][cols_impute])
        
        print(f"Median-imputed {len(cols_impute)} columns in '{name}'.")
    
    return dfs

# --------------------------------------------------------

def fix_encoded_cols(dfs, ignore_cols=None):
    """
    Ensures all dataframes in dfs have the same columns after encoding. If a DataFrame is missing a column that exists in others, we add it, filled with 0.
    
    Args:
        dfs (dict): Dictionary of dataframes.
        ignore_cols (list): Columns to ignore (e.g. ['RESPONSE']) when fixing encoded cols.
    
    Returns:
        dfs (dicts): Updated dictionary of dataframes with fixed columns.
    """

    if ignore_cols is None:
        ignore_cols = []
    
    # Gather all cols
    all_columns = set()
    for name in dfs.keys():
        cols = set(dfs[name].columns) - set(ignore_cols)
        all_columns |= cols  
    
    # For each DataFrame, find which columns are missing
    for name in dfs.keys():
        current_cols = set(dfs[name].columns) - set(ignore_cols)
        missing_cols = all_columns - current_cols
        if missing_cols:
            # Add missing columns filled with 0
            print(f"'{name}' is missing {len(missing_cols)} columns. Adding with zeres imputed.")
            missing_df = pd.DataFrame(0, index=dfs[name].index, columns=list(missing_cols))
            dfs[name] = pd.concat([dfs[name], missing_df], axis=1)
        # Reordering, and place ignore_cols (like 'RESPONSE') at the end if they exist.
        final_col_order = sorted(all_columns) + sorted(set(dfs[name].columns) & set(ignore_cols))
        dfs[name] = dfs[name][final_col_order]
    
    return dfs

# --------------------------------------------------------

def set_plotting_style():
    """
    Sets global plotting style.
    """
    sns.set_theme(style="whitegrid") 
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        'skyblue', 'salmon', 'mediumseagreen', 'gold', 'mediumslateblue', 
        'darkorange', 'orchid', 'lightcoral'])
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['grid.color'] = 'lightgray'