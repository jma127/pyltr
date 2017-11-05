import pandas as pd


class PandasLetorConverter(object):
    '''
    Class Converter implements parsing from original MSLR-WEB and LETOR
    txt files to pandas data frame representation.
    '''

    def __init__(self, path):
        '''
        Arguments:
            path (str): path to letor txt file
        '''
        self.path = path


    @property
    def path(self):
        return self._path


    @path.setter
    def path(self, p):
        if type(p) is not str:
            raise TypeError('path must be of type str')
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

        tracker = 0

        # Ensures compatibility with MSLR-WEBK datasets
        for col in range(1,len(df.columns)):
            tracker += 1
            if ':' in str(df.ix[:,col][0]):
                df.ix[:,col] = df.ix[:,col].apply(lambda x: str(x).split(':')[1])
            else:
                break
                #tracker = col

        df.columns = ['rel', 'qid'] + [str(x) for x in range(1,len(df.columns)-1)] # renaming cols

        # Ensures compatibility with LETOR datasets
        if tracker != len(df.columns)-1:
            newcols = []
            for col in df.columns:
                test = df.ix[0,col]
                if ('docid' in str(test)) or ('inc' in str(test)) or ('prob' in str(test)) or ('=' in str(test)):
                    newcols.append(test)
                    df = df.drop(str(col), axis=1)
            newcols = [x for x in newcols if '=' not in x]
            df.columns.values[-len(newcols):] = newcols

        return df


    def convert(self):
        '''
        Performs final conversion.
        Return:
            fully converted pandas dataframe
        '''
        df_raw = self._load_file()
        df_drop = self._drop_col(df_raw)
        return self._split_colon(df_drop)