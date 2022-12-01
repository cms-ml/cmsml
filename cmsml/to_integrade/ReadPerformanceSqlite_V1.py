import sqlite3
from typing import Sequence, Union
from pathlib import Path
import numpy as np


class SQL3Reader():
    def __init__(self):
        self.db = None  # self.connect(database_path)
        self.open_connection = False
        self.database_location = None

    @staticmethod
    def green(string):
        return f'\033[1;32m {string} \033[0m'

    @staticmethod
    def red(string):
        return f'\033[0;31m {string} \033[0m'

    def connect(self, path_database: str):
        """
        Creates connection to database and set status flags to True"""
        db_path = Path(path_database).with_suffix('.sql3')
        # to prevent python from creating a sql database
        # sqlite3 creates an DB if none exists, instead raises now an error
        if not db_path.exists():
            raise ValueError(f'Path: {db_path} does not exists.')

        msg = f'Established connection to: \n \"{db_path}\"'
        print(self.green(msg))
        self.open_connection = True
        self.database_location = str(db_path)
        self.db = sqlite3.connect(str(db_path))

    def close(self):
        """
        Close database connection and set status flags to false
        """
        msg = f'Close connection to: \n \"{self.database_location}\"'
        print(self.red(msg))
        self.db.close()
        self.open_connection = False

    def execute_cursor(self, cmd: str):
        """
        Shortcut for one of the most common cursor execution

        Args:
            cmd (str): SQL command

        Returns:
            sqlite3.cursor: Cursor of the db with the command. To actually
            get data use fetchall()
        """
        cursor = self.db.cursor()
        cursor.execute(cmd)
        return cursor

    def print_database_columns(self):
        """
        Reads all columns from SQL Table and prints them in a table
        """
        cursor = self.execute_cursor(
            "SELECT name FROM sqlite_master WHERE type='table';")
        result = cursor.fetchall()
        # extract table names
        table_names = sorted(list(zip(*result))[0])
        print('\n \n Listing all Tables and Columns')
        for table_name in table_names:
            re = cursor.execute("PRAGMA table_info('%s')" %
                                table_name).fetchall()
            column_names = '\n \t | '.join(list(zip(*re))[1])
            print('--' * 20)
            print(f'\t {table_name} \n \t | {column_names}')
        print('--' * 20, '\n' * 2)


class IgProfReader(SQL3Reader):

    column_names = {
        'mainrow': ('id', 'symbol_id', 'self_count', 'cumulative_count', 'kids', 'self_calls', 'total_calls', 'self_paths', 'total_paths', 'pct'),
        'children': ('self_id', 'parent_id', 'from_parent_count', 'from_parent_calls', 'from_parent_paths', 'pct'),
        'files': ('id', 'name'),
        'parents': ('self_id', 'child_id', 'to_child_count', 'to_child_calls', 'to_child_paths', 'pct'),
        'summary': ('counter', 'total_count', 'total_freq', 'tick_period'),
        'symbols': ('id', 'name', 'filename_id')}

    def __init__(self):
        super().__init__()

    @classmethod
    def convert_bytes(cls, number: Union[Sequence, float, int], suffix: str = 'B', round_to: int = 10, numeric: bool = True):
        number = np.array(number)
        number = IgProfReader._convert_bytes(number, suffix, round_to, numeric)
        return number

    @staticmethod
    def _convert_bytes(number: Union['np.array', float, int], suffix: str = 'B', round_to: int = 10, numeric: bool = True) -> Union[float, str]:
        """
        Converts bytes to the unit in suffix.
        By Default rounds to 10 deczimal precision if numeric is on a str is returned with the correct unit

        Args:
            number (Union[float, int]): Number u want to convert
            suffix (str, optional): Conversion suffix
            round_to (int, optional): number of decimal precision
            numeric (bool, optional): return the number or a string with the unit.

        Returns:
            Union[float, str]: Str or Number
        """
        BINARY_LABELS = {'B': 0, 'KiB': 1, 'MiB': 2, 'GiB': 3,
                         'TiB': 4, 'PiB': 5, 'EiB': 6, 'ZiB': 7, 'YiB': 8}

        if suffix in BINARY_LABELS:
            potence = 1024.0 ** BINARY_LABELS[suffix]
        number = number / potence
        number = round(number, round_to)

        if numeric:
            return number
        return f'{number} {suffix}'

    @classmethod
    def print_igprof_columns(cls):
        """
        Prints all columns from IgProf Sqlite DB out. To get an overview of all columns
        """
        for header, columns in IgProfReader.column_names.items():
            column_names = '\n\t| '.join(columns)
            print(f'Header: {header} \n \t| {column_names}')

    def get_column_name(self, key: str):
        """
        Args:
            key (str): column name (see dictionary: column_names)

        Returns:
            tuple(str): tuple of children columns

        Raises:
            KeyError: If wrong key is passed!

        """
        if key in self.column_names.keys():
            return self.column_names.get(key)
        else:
            existing_keys = ', '.join(list(db.column_names.keys()))
            raise KeyError(f'Key: {key} does not exist. Existing keys are: {existing_keys}')

    def filter_process_in_column(self, column: Union[Sequence[str], str], processes: Union[Sequence[str], str]):
        """
        Combines multiple 'column LIKE process[0] OR column LIKE process[1] OR ....
        E.g: filter_process_in_column('name','Aot%', 'Perf%') will result in
        name LIKE Aot% OR name LIKE Perf%.

        Main usage is to filter out multiple values within a column

        Args:
            column (Union[Sequence[str], str]): Sequence of column names
            processes (Union[Sequence[str], str]): Description

        Returns:
            str: cmd describin the LIKES_ORs
        """
        # combine columns and processes with an LIKE
        likes = [f'{column} LIKE \"{process}\" ' for process in processes]
        query = ' OR '.join(likes)
        return query

    def chain(self, sequence: Sequence[str]):
        """
        Join sequences of strings with ,.
        Example: ("symbols", "id", "name") -> "symbols, id, name"
        Ready to be used in the query

        Args:
            sequence (Sequence[str]): Sequence of strings

        Returns:
            str: all strings joined with ,
        """
        if isinstance(sequence, (tuple, list, set)):
            return ', '.join(sequence)
        return sequence

    def selection_from_mainrow(self: Union[Sequence, str], columns: Union[Sequence, str], *processes: Union[Sequence, str]):
        """
        Gets <columns> values of a specific <processes> the "mainrow" table:
        The columns need to be a sequence of strings or a string.
        The processes can use wildcards (%,_).

        Args:
            columns (Union[Sequence, str]): Description
            *processes (Union[Sequence, str]): Description

        Returns:
            sqlite3.cursor: Returns selection with sqlite3.cursor. To get data use cursor.fetchall()
        """

        # create column strings
        columns = self.chain(columns)
        # WHERE condition: name LIKE process[0] OR name LIKE process[1]
        like_condition_chain = self.filter_process_in_column('name', processes)

        result_query = f'''SELECT {columns}
                           FROM symbols
                           LEFT JOIN mainrows ON mainrows.symbol_id = symbols.id
                           WHERE {like_condition_chain}'''
        results = self.execute_cursor(result_query)
        return results

    def get_AOT_data(self, column: str, process: str = 'void AotUtilis::loadMultipleAotModels<AOT_BATCH_SIZE_%>(int)'):
        """
        Get from the table mainrow the function values of all <columns> and the
        batchsize as numpy arrays.

        The FUNCTION NEEDS TO BE NAMED IN A SPECIFIC WAY

        Returns 2 numpy arrays to plot with matplotlib.
        """
        column = self.chain(column)
        names = self.selection_from_mainrow('name', process).fetchall()

        # extract batch sizes from Name in the same ordner
        batch_sizes = np.array(
            [int(name[0].replace('>(int)', '').split('_')[-1]) for name in names])

        selection = self.selection_from_mainrow(column, process)
        # data is of forem: [(column1 column2 ...), (), ()], with () being a row
        data = selection.fetchall()

        data = np.array(data).flatten().reshape(len(batch_sizes), -1)
        return batch_sizes, data

    @staticmethod
    def flatten_selection(list):
        return [item for sublist in list for item in sublist]

    @staticmethod
    def create_default_memory():
        from collections import defaultdict
        basket = defaultdict(lambda: defaultdict(list))
        return basket

    @staticmethod
    def assemble_sql3_path(location_of_sqlfiles: str, sql_profile_names: Union[Sequence[str], str], mode: str) -> str:
        """
        String operations where <location_of_sqlfiles> is joint with
        <sql_profile_names>, where the compression is replaced with
        the <mode>.sql3 of all sqlfiles are

        Args:
            location_of_sqlfiles (str): path to directory of sqlfiles
            sql_profile_names (Union[Sequence[str], str]): name of the IgProf profile
            mode (str): MEM_LIVE, MEM_MAX, MEM_LIVE_PEAK or MEM_TOTAL

        Returns:
            str: absolute path to the sql3 report, ready to open.
        """
        def replace_suffix(sql3_paths: Sequence[str], replace_with: str, old_suffix: str = '.gz'):
            """
            Args:
                sql3_paths (Sequence[str]): Paths to sql3 dbs
                replace_with (str): string you want to insert (like MEM_LIVE)
                old_suffix (str, optional): string to you want to replace (default: '.gz')

            Returns:
                TYPE: Description

            """
            sql_suffix = 'sql3'
            final_suffix = '.'.join(('', replace_with, sql_suffix))
            replaced_sq3_paths = [sql3_path.replace(old_suffix, final_suffix)
                                  for sql3_path in sql3_paths]
            return replaced_sq3_paths

        sql3_files = replace_suffix(sql_profile_names, mode.upper())
        p = Path(location_of_sqlfiles)
        sql3_files = [str(p.joinpath(sql3)) for sql3 in sql3_files]
        return sql3_files

    @staticmethod
    def aot_string(batch_size):
        return ''.join(('void AotUtilis::loadMultipleAotModels<AOT_BATCH_SIZE_', str(batch_size), '>(int)'))

    @classmethod
    def get_column_from_multiple_tables(cls, path_sql3_files: Sequence[str], column_names: str):
        temp_db = IgProfReader()
        basket = temp_db.create_default_memory()
        for sql_db in path_sql3_files:
            temp_db.connect(sql_db)
            for function_name, value in column_names.items():
                values = value.split(',')
                for v in values:
                    v = v.strip(' ')
                    data = temp_db.selection_from_mainrow(v, function_name).fetchall()
                    if not data:
                        warning = 'return value for: {function_name}, is empty. Please check if your query is written right'
                        print(IgProfReader.red(warning))
                        continue
                    data = temp_db.flatten_selection(data)[0]
                    basket[function_name][v].append(data)
        temp_db.close()
        return basket


if __name__ == '__main__':

    # path = '/nfs/dust/cms/user/wiedersb/igprof_logs/sql3/TF_1_really_correct.mp.gz_MEM_TOTAL.sql3'
    # path = '/nfs/dust/cms/user/wiedersb/igprof_logs/sql3/AOT_LOAD1_ALLBATCH.sql3'
    path = '/nfs/dust/cms/user/wiedersb/igprof_logs/sql3/copy_to/AOT_LOAD-1-MODEL_STATIC.mp.MEM_TOTAL.sql3'
    path = '/nfs/dust/cms/user/wiedersb/igprof_logs/sql3/copy_to/AOT_LOAD-1-MODEL_RUNS-200_WARM-50_DYNAMIC.mp.MEM_TOTAL.sql3'
    # path = '/nfs/dust/cms/user/wiedersb/igprof_logs/sql3/10_TF_240_128_SEASON-1.mp.MEM_TOTAL.sql3'
    # path = '/nfs/dust/cms/user/wiedersb/igprof_logs/sql3/allocation_test_graph_recreation.mp.MEM_TOTAL.sql3'

    sql_suffix = '.sql3'

    AOT_12L_128N = ('001_AOT_12L_128N_1M_ALLB.mp.gz', '002_AOT_12L_128N_2M_ALLB.mp.gz',
                    '003_AOT_12L_128N_3M_ALLB.mp.gz', '004_AOT_12L_128N_4M_ALLB.mp.gz')
    AOT_25L_128N = ('005_AOT_12L_128N_1M_ALLB.mp.gz', '006_AOT_12L_128N_2M_ALLB.mp.gz',
                    '007_AOT_12L_128N_3M_ALLB.mp.gz', '008_AOT_12L_128N_4M_ALLB.mp.gz')

    TF_12L_128N = ('001_TF_12L_128N_1S.mp.gz', '002_TF_12L_128N_2S.mp.gz',
                   '003_TF_12L_128N_3S.mp.gz', '004_TF_12L_128N_4S.mp.gz')

    db = IgProfReader()
    db.connect(path)
    db.print_database_columns()

    data = db.selection_from_mainrow(
        # 'name, cumulative_count, self_count, self_calls', '_xla_aot_model_batch_%_AOT_BATCH_SIZE_%')
        'name, cumulative_count, self_count, self_calls', 'AotUtilis%', 'Perf%')

    from IPython import embed
    embed()
    # def get_process_profiles_DEPRECIATED(self, process_name: str):
    #     """
    #     returns the values of the "mainrow" table based on its symbolic id from the table "symbols".
    #     The <process_name> can be either the exact name of the process
    #      or a template using the wildcards *, _.

    #     The return value is a named_tuple of the mainrow arguments.

    #     This function is depreciated and replaced with a more MySQL - like solution.
    #     The only reason this still exists is because this solution is more educative.
    #     """
    #     symbols_query = f"SELECT DISTINCT id, name FROM symbols WHERE name LIKE \"{process_name}\""
    #     symbols_row = self.execute_cursor(symbols_query)

    #     results = {}
    #     for symbols_id, symbols_name in symbols_row:
    #         mainrow_query = f"SELECT mainrows.* FROM mainrows WHERE mainrows.symbol_id=\"{str(symbols_id)}\""
    #         mainrow_table = self.execute_cursor(mainrow_query)
    #         results[symbols_name] = mainrow_table.fetchone()
    #     return results

    # def get_process_profiles_DEPRECIATED_2(self, process_name: str, verbose=False):
    #     """
    #     Gets values of the "mainrow" table based on its symbolic id from the table "symbols".
    #     The return value is a dictionary with the key being the name of the process . e.g.
    #     PerfTester::loadedSeassion().

    #     The values stores with this key is a named tuple. with the fields being the names of the row.
    #     The reason for this structure is that namedtuple namings are quiet restricted, (for example brackets or even :: are forbidden.)

    #     <process name> : can be either the exact name of the process
    #      or a template using the wildcards *, _.
    #      <verbose> : boolean to print name of the current processing function
    #     """
    #     column_name = "mainrow"
    #     names_query = f"SELECT name FROM symbols WHERE name LIKE \"{process_name}\""
    #     result_names = self.execute_cursor(names_query)

    #     query = f"SELECT * FROM mainrows WHERE symbol_id IN (SELECT id FROM symbols WHERE name LIKE \"{process_name}\")"
    #     result = self.execute_cursor(query)

    #     result_ = {}
    #     for i, (name, values) in enumerate(zip(result_names, result)):
    #         # name is a tuple (value,)
    #         name = name[0]
    #         if verbose:
    #             print(f'Get function: {name}')

    #         named_result = namedtuple(
    #             'data'+str(i), self.column_names[column_name])
    #         filled_named_result = named_result(*values)
    #         result_[name] = filled_named_result

    #     return result_

    # def wrap_in_tuple(self, input):
    #     if not isinstance(input, tuple, list):
    #         return (input)
    #     else:
    #         return input
