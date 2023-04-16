#
#
# (c) Ricky Macharm, MScFE
# https://SisengAI.com
#
#

import os
import pandas as pd
import numpy as np
import urllib.request
import zipfile

from feature_engine import encoding, imputation
from sklearn import base, pipeline

from sklearn import model_selection

import xgboost as xgb

import subprocess

def extract_dataset(src_path: str, dst_path: str, dataset: str) -> pd.DataFrame: 
    """Extracts a chosen dataset from a zip file and read it into a pandas
    DataFrame.
    
    Parameters:
    ------------
    src_path: str
        URL or path of the zip file to be downloaded.
    dst_path: str
        Local file path where the zip file will be written.
    dataset: str
        Name of the particular file inside the zip file to be 
        read into a DataFrame.
    
    Returns:
    ---------
    pandas.core.frame.DataFrame: 
        DataFrame containing the contents of the selected dataset.
        
    """
    
    # using `urllib.request` module to open the URL specified in `src_path`.    
    open_path = urllib.request.urlopen(src_path)
    
    # reads the contents of the file object `open_path`
    data = open_path.read()
    
    # using context management to write the file in the destination path or folder
    with open(dst_path, mode='wb') as f:
        f.write(data)
       
    # using context management to extract the saved zip file.
    with zipfile.ZipFile(dst_path) as z:
        all_df = pd.read_csv(z.open(dataset))
        df_ = all_df.iloc[0]
        df = all_df.iloc[1:]
        return df
    
    

def topx(pser: pd.Series, x: int = 5, default='other') -> pd.Series:
    """Replace all values in a Pandas Series that are not among
    the top `x` most frequent values with a default value.

    Parameters
    ----------
    pser : pd.Series
        The input pandas Series.
    x : int, optional
        The number of most frequent values to keep. The
        default value is 5.
    default : str, optional
        The default value to use for values that are not among
        the top `x` most frequent values. The default value is
        'other'.

    Returns:
    ---------
    pandas.Series: 
        Series containing the results.
    """

    val_count = pser.value_counts()
    return (pser.where(pser
                      .isin(val_count.index[:x]),
                      default)
           )


def prep_data(pdf: pd.DataFrame) -> pd.DataFrame:
    """ This function prepares the rurvey data and returns a new DataFrame.
    The prepatations include extracting and transforming certain
    columns, renaming some, and selecting a subset of columns.
    
    Parameters
    ----------
    pdf : pd.DataFrame
        The input DataFrame containing the survey data.
    
    Returns
    -------
    pd.DataFrame:
        The new DataFrame with the modified and selected columns.
    """
    
    return (pdf
            .assign(age=(pdf['Q2']
                         .str.slice(0,2)
                         .astype(int)
                        ),
                   education=(pdf
                             ['Q4']
                             .replace({'Doctoral degree' : 20, 'Bachelor’s degree' : 16, 
                                       'Master’s degree' : 18,
                                   'Professional degree' : 19,
                                   'Some college/university study without earning a bachelor’s degree': 13,
                                   'I prefer not to answer' : None, 
                                    'No formal education past high school': 12})
                            ),
                    major=(pdf['Q5']
                           .pipe(topx, x=3)
                           .replace({'Engineering (non-computer focused)': 'eng',
                                       'Computer science (software engineering, etc.)': 'cs',
                                       'Mathematics or statistics' : 'stat',})
                          ),
                    years_exp=(pdf
                             ['Q8']
                             .str.replace('+', '', regex=False)
                             .str.split('-', expand=True)
                             .iloc[:,0]
                             .astype(float)
                            ),
                    compensation=(pdf
                             ['Q9']
                             .str.replace('+', '', regex=False)
                             .str.replace(',','')
                             .replace({'500000':'500',
                                      'I do not wish to disclose my approximate yearly compensation':'0'})
                             .str.split('-', expand=True)
                             .iloc[:,0]
                             .fillna(0)
                             .astype(int)
                             .mul(1_000)
                            ),
                    python=(pdf
                             ['Q16_Part_1']
                             .fillna(0)
                             .replace({'Python': 1})
                            ),
                    r=(pdf
                         ['Q16_Part_2']
                         .fillna(0)
                         .replace({'R': 1})
                        ),
                    sql=(pdf
                         ['Q16_Part_3']
                         .fillna(0)
                         .replace({'SQL': 1})
                        )
                    )
            .rename(columns=lambda col:col.replace(' ', '_'))
            .loc[:, 'Q1,Q3,age,education,major,years_exp,compensation,python,r,sql'.split(',')]
    )


class PrepDataTransformer(base.BaseEstimator,
    base.TransformerMixin):
    """
    This transformer takes a Pandas DataFrame containing our survey 
    data as input and returns a new version of the DataFrame. 
    
    ----------
    ycol : str, optional
        The name of the column to be used as the target variable.
        If not specified, the target variable will not be set.
    Attributes
    ----------
    ycol : str
        The name of the column to be used as the target variable.
    """
    def __init__(self, ycol=None):
        self.ycol = ycol
    
    def transform(self, X):
        return prep_data(X)

    def fit(self, X, y=None):
        return self

def prepX_y(df, y_col):
    raw = (df
    .query('Q3.isin(["United States of America", "China", "India"]) '
    'and Q6.isin(["Data Scientist", "Software Engineer"])')
    )
    return raw.drop(columns=[y_col]), raw[y_col]


def my_dot_export(model, n_trees, filename, title='', direction='TB'):
    """Exports a specified number of trees from an XGBoost model as a graph
    visualization in dot and png formats.
    Parameters:
    -----------
    model: 
        An XGBoost model.
    n_trees: 
        The number of tree to export.
    filename: 
        The name of the file to save the exported visualization.
    title: 
        The title to display on the graph visualization (optional).
    direction: 
        The direction to lay out the graph, either 'TB' (top to
        bottom) or 'LR' (left to right) (optional).
    """
    res = xgb.to_graphviz(model, num_trees=n_trees)
    content = f''' node [fontname = "Roboto Condensed"];
    edge [fontname = "Roboto Thin"];
    label = "{title}"
    fontname = "Roboto Condensed"
    '''
    out = res.source.replace('graph [ rankdir=TB ]',
                             f'graph [ rankdir={direction} ];\n {content}')
    # dot -Gdpi=300 -Tpng -ocourseflow.png courseflow.dot
    dot_filename = filename
    with open(dot_filename, 'w') as f:
        f.write(out)
    png_filename = dot_filename.replace('.dot', '.png')
    subprocess.run(f'dot -Gdpi=300 -Tpng -o{png_filename} {dot_filename}'.split())
    
    
def inv_logit(p: float) -> float:
    """
    Compute the inverse logit function of a given value.
    The inverse logit function is defined as:
    f(p) = exp(p) / (1 + exp(p))
    Parameters
    ----------
    p : float
        The input value to the inverse logit function.
    
    Returns
    -------
    float
        The output of the inverse logit function.
    """
    return np.exp(p) / (1 + np.exp(p))