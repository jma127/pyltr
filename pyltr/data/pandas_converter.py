import pandas as pd


class PandasLetorConverter(object):
    '''
    Class Converter implements parsing from original letor txt files to
    pandas data frame representation.
    '''

    def __init__(self, path):
        '''
        Arguments:
            path: path to letor txt file
        '''
        self._path = path

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, p):
        self._path = p

    def _load_file(self):
        '''
        Loads and parses raw letor txt file.

        Return:
            letor txt file parsed to csv in raw format
        '''
        return pd.read_csv(str(self._path), sep=" ", header=None)

    def _drop_col(self, df):
        '''
        Drops last column, which was added in the parsing procedure due to a
        trailing white space for each sample in the text file

        Arguments:
            df: pandas dataframe
        Return:
            df: original df with last column dropped
        '''
        return df.drop(df.columns[-1], axis=1)

    def _split_colon(self, df):
        '''
        Splits the data on the colon and transforms it into a tabular format
        where columns are features and rows samples. Cells represent feature
        values per sample.

        Arguments:
            df: pandas dataframe object
        Return:
            df: original df with string pattern ':' removed; columns named appropriately
        '''
        for col in range(1,len(df.columns)):
            df.loc[:,col] = df.loc[:,col].apply(lambda x: str(x).split(':')[1])
        df.columns = ['rel', 'qid'] + [str(x) for x in range(1,len(df.columns)-1)] # renaming cols
        return df

    def convert(self):
        '''
        Performs final conversion.

        Return:
            fully converted pandas dataframe
        '''
        df_raw = self._load_file(self._path)
        df_drop = self._drop_col(df_raw)
        return self._split_colon(df_drop)
