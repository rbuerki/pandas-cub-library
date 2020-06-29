from typing import Dict, List, Tuple
import numpy as np

__version__ = "0.0.1"


class DataFrame:
    def __init__(self, data: Dict) -> None:
        """A DataFrame holds tabular data orginzed by index and columns.
        Create it by passing a dictionary of NumPy arrays to the values
        parameter.

        Args:
            data: A dictionary of strings mapped to NumPy arrays.
            The keys will become the column names.
        """
        # check for correct input types
        self._check_input_types(data)

        # check for equal array lengths
        self._check_array_lengths(data)

        # convert unicode arrays to object
        self._data = self._convert_unicode_to_object(data)

        # Allow for special methods for strings
        self.str = StringMethods(self)
        self._add_docs()

    def _check_input_types(self, data: Dict) -> None:
        """Perform several checks to make sure that the input data is
        in the right format. Raise errors if not.

        Args:
            data: A dictionary of strings mapped to NumPy arrays.

        Raises:
            TypeError: If `data`is not of type dict.
            TypeError: If keys of `data` are not of type string.
            TypeError: If values of `data` are not of type np.ndarray.
            ValueError: If the arrays are not one-dimensional.
        """
        if not isinstance(data, dict):
            raise TypeError("`data` must be a dictionary.")
        for key, value in data.items():
            if not isinstance(key, str):
                raise TypeError("Keys of `data` must be strings.")
            if not isinstance(value, np.ndarray):
                raise TypeError("Values of `data`must be numpy arrays.")
            if value.ndim != 1:
                raise ValueError(
                    "Values of `data` must be a one-dimensional array."
                )

    def _check_array_lengths(self, data: Dict) -> None:
        """Perform a check to make sure that the all arrays in `data`
        the same length. Raise error if not.

        Args:
            data: A dictionary of strings mapped to NumPy arrays.

        Raises:
            ValueError: If the arrays are not the same length.
        """
        for i, value in enumerate(data.values()):
            if i == 0:
                length = len(value)
            elif length != len(value):
                raise ValueError(
                    " All values of `data` must be the same length."
                )

    def _convert_unicode_to_object(self, data: Dict) -> Dict:
        """Change the data type of Unicode arrays in `data` to object.
        Return a new dict with the transformed data.

        Args:
            data: A dictionary of strings mapped to NumPy arrays.

        Returns:
            Dict: A dictionary where the data type for string arrays
            has been changed from unicode to object.

        Note:
            Whenever you create a numpy array of Python strings, it will
            default the data type of that array to unicode. Take a look at
            the following simple numpy array created from strings. Its
            data type, found in the `dtype` attribute is shown to be 'U'
            plus the length of the longest string.

            ```python
            >>> a = np.array(['cat', 'dog', 'snake'])
            >>> a.dtype
            dtype('<U5')
            ```

            Unicode arrays are more difficult to manipulate and don't
            have the flexibility that we desire. So, if our user
            passes us a Unicode array, we will convert it to a data type
            called 'object'. This is a flexible type and will help us
            later when creating methods just for string columns. Technically,
            this data type allows any Python objects within the array.

            In this funciton we do this by checking each arrays data type `kind`.
            The data type `kind` is a single-character value available by
            doing `array.dtype.kind`. See the [numpy docs][8]
            for a list of all the available kinds.

        Additional info:
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.kind.html#numpy.dtype.kind  # noqa: B950
        """
        new_data = {}
        for key, value in data.items():
            if value.dtype.kind == "U":
                value = value.astype("O")
            new_data[key] = value
        return new_data

    def __len__(self):
        """Make the builtin len function work with our dataframe. To do so
        we need to implement the special method `__len__`. This is what
        Python calls whenever an object is passed to the `len` function.

        Returns:
            int: The number of rows in the dataframe.

        Note:
            Python has over 100 special methods that allow you to define
            how your class behaves when it interacts with a builtin
            function or operator. In the above example, if `df` is a
            DataFrame and a user calls `len(df)` then internally the
            `__len__` method will be called. All special methods begin
            and end with two underscores.

            Let's see a few more examples:

            * `df + 5` calls the `__add__` special method
            * `df > 5` calls the `__lt__` special method
            * `-df` calls the `__neg__` special method
            * `round(df)` calls the `__round__` special method

            We've actually already seen the special method `__init__`
            which is used to initialize an object and called when a
            user calls `DataFrame(data)`.

        Additional info:
            https://docs.python.org/3/reference/datamodel.html#specialnames
        """
        return len(next(iter(self._data.values())))

    @property
    def columns(self) -> List[str]:
        """
        Retrieve the column names from `_data` and return a list of the
        columns in order. (Only works in Python 3.6+)

        Returns:
            A list of column names.

        Note:
            Notice that `df.columns` is not a method here. (There are no
            parentheses that follow it. The `property` decorator will turn
            `df.columns` into an attribute (that works just like a method).

            There are three parts to properties in Python; the getter, setter,
            and deleter. This is the getter. The next function defines the
            setter. (We will not implement the deleter.)

        Additional info:
            https://stackoverflow.com/questions/17330160/how-does-the-property-decorator-work  # noqa: B950

        """
        return list(self._data)

    @columns.setter
    def columns(self, columns: List[str]):
        """Set new column names from a list and update the property of the
        actual dataframe.

        Args:
            columns: List of column names.

        Raises:
            TypeError: If the object used to set new columns is not a list.
            ValueError: If the number of column names in the list does
                not match the current DataFrame.
            TypeError: If any of the columns are not strings.
            ValueError: If any of the column names are duplicated in the list.
        """
        if not isinstance(columns, list):
            raise TypeError("`columns` must be a list.")
        if not len(columns) == len(self._data):
            raise ValueError(
                "`columns` must be of same length as the current dataframe."
            )
        for col in columns:
            if not isinstance(col, str):
                raise TypeError("All values of `columns` must be strings.")
        if len(set(columns)) != len(columns):
            raise ValueError("Alle values of `columns` must be unique.")
        self._data = dict(zip(columns, self._data.values()))

    @property
    def shape(self) -> Tuple[int, int]:
        """Return a two-item tuple of the number of rows and columns."""
        return (
            len(self),
            len(self._data),
        )

    def _repr_html_(self):
        """
        Create a string of HTML to nicely display the DataFrame in a
        Jupyter Notebook. 

        Note:
            The `_repr_html_` method is made available to developers
            by iPython so that your objects can have nicely formatted
            HTML displays within Jupyter Notebooks.

        More information:
            https://ipython.readthedocs.io/en/stable/config/integrating.html
        """
        html = "<table><thead><tr><th></th>"
        for col in self.columns:
            html += f"<th>{col:10}</th>"

        html += "</tr></thead>"
        html += "<tbody>"

        only_head = False
        num_head = 10
        num_tail = 10
        if len(self) <= 20:
            only_head = True
            num_head = len(self)

        for i in range(num_head):
            html += f"<tr><td><strong>{i}</strong></td>"
            for _col, values in self._data.items():
                kind = values.dtype.kind
                if kind == "f":
                    html += f"<td>{values[i]:10.3f}</td>"
                elif kind == "b":
                    html += f"<td>{values[i]}</td>"
                elif kind == "O":
                    v = values[i]
                    if v is None:
                        v = "None"
                    html += f"<td>{v:10}</td>"
                else:
                    html += f"<td>{values[i]:10}</td>"
            html += "</tr>"

        if not only_head:
            html += "<tr><strong><td>...</td></strong>"
            for _i in range(len(self.columns)):
                html += "<td>...</td>"
            html += "</tr>"
            for i in range(-num_tail, 0):
                html += f"<tr><td><strong>{len(self) + i}</strong></td>"
                for _col, values in self._data.items():
                    kind = values.dtype.kind
                    if kind == "f":
                        html += f"<td>{values[i]:10.3f}</td>"
                    elif kind == "b":
                        html += f"<td>{values[i]}</td>"
                    elif kind == "O":
                        v = values[i]
                        if v is None:
                            v = "None"
                        html += f"<td>{v:10}</td>"
                    else:
                        html += f"<td>{values[i]:10}</td>"
                html += "</tr>"

        html += "</tbody></table>"
        return html

    @property
    def values(self) -> np.array:
        """Return a single 2D NumPy array of the underlying data."""
        return np.column_stack(list(self._data.values()))

    @property
    def dtypes(self):
        """
        Returns
        -------
        A two-column DataFrame of column names in one column and
        their data type in the other
        """
        # DTYPE_NAME = {"O": "string", "i": "int", "f": "float", "b": "bool"}
        pass

    def __getitem__(self, item):
        """
        Use the brackets operator to simultaneously select rows and columns
        A single string selects one column -> df['colname']
        A list of strings selects multiple columns -> df[['colname1', 'colname2']]
        A one column DataFrame of booleans that filters rows -> df[df_bool]
        Row and column selection simultaneously -> df[rs, cs]
            where cs and rs can be integers, slices, or a list of integers
            rs can also be a one-column boolean DataFrame

        Returns
        -------
        A subset of the original DataFrame
        """
        pass

    def _getitem_tuple(self, item):
        # simultaneous selection of rows and cols -> df[rs, cs]
        pass

    def _ipython_key_completions_(self):
        # allows for tab completion when doing df['c
        pass

    def __setitem__(self, key, value):
        # adds a new column or a overwrites an old column
        pass

    def head(self, n=5):
        """
        Return the first n rows

        Parameters
        ----------
        n: int

        Returns
        -------
        DataFrame
        """
        pass

    def tail(self, n=5):
        """
        Return the last n rows

        Parameters
        ----------
        n: int

        Returns
        -------
        DataFrame
        """
        pass

    # ### Aggregation Methods ####

    def min(self):
        return self._agg(np.min)

    def max(self):
        return self._agg(np.max)

    def mean(self):
        return self._agg(np.mean)

    def median(self):
        return self._agg(np.median)

    def sum(self):
        return self._agg(np.sum)

    def var(self):
        return self._agg(np.var)

    def std(self):
        return self._agg(np.std)

    def all(self):
        return self._agg(np.all)

    def any(self):
        return self._agg(np.any)

    def argmax(self):
        return self._agg(np.argmax)

    def argmin(self):
        return self._agg(np.argmin)

    def _agg(self, aggfunc):
        """
        Generic aggregation function that applies the
        aggregation to each column

        Parameters
        ----------
        aggfunc: str of the aggregation function name in NumPy

        Returns
        -------
        A DataFrame
        """
        pass

    def isna(self):
        """
        Determines whether each value in the DataFrame is missing or not

        Returns
        -------
        A DataFrame of booleans the same size as the calling DataFrame
        """
        pass

    def count(self):
        """
        Counts the number of non-missing values per column

        Returns
        -------
        A DataFrame
        """
        pass

    def unique(self):
        """
        Finds the unique values of each column

        Returns
        -------
        A list of one-column DataFrames
        """
        pass

    def nunique(self):
        """
        Find the number of unique values in each column

        Returns
        -------
        A DataFrame
        """
        pass

    def value_counts(self, normalize=False):
        """
        Returns the frequency of each unique value for each column

        Parameters
        ----------
        normalize: bool
            If True, returns the relative frequencies (percent)

        Returns
        -------
        A list of DataFrames or a single DataFrame if one column
        """
        pass

    def rename(self, columns):
        """
        Renames columns in the DataFrame

        Parameters
        ----------
        columns: dict
            A dictionary mapping the old column name to the new column name

        Returns
        -------
        A DataFrame
        """
        pass

    def drop(self, columns):
        """
        Drops one or more columns from a DataFrame

        Parameters
        ----------
        columns: str or list of strings

        Returns
        -------
        A DataFrame
        """
        pass

    # ### Non-Aggregation Methods ####

    def abs(self):
        """
        Takes the absolute value of each value in the DataFrame

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.abs)

    def cummin(self):
        """
        Finds cumulative minimum by column

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.minimum.accumulate)

    def cummax(self):
        """
        Finds cumulative maximum by column

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.maximum.accumulate)

    def cumsum(self):
        """
        Finds cumulative sum by column

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.cumsum)

    def clip(self, lower=None, upper=None):
        """
        All values less than lower will be set to lower
        All values greater than upper will be set to upper

        Parameters
        ----------
        lower: number or None
        upper: number or None

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.clip, a_min=lower, a_max=upper)

    def round(self, n):
        """
        Rounds values to the nearest n decimals

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.round, decimals=n)

    def copy(self):
        """
        Copies the DataFrame

        Returns
        -------
        A DataFrame
        """
        return self._non_agg(np.copy)

    def _non_agg(self, funcname, **kwargs):
        """
        Generic non-aggregation function

        Parameters
        ----------
        funcname: numpy function
        kwargs: extra keyword arguments for certain functions

        Returns
        -------
        A DataFrame
        """
        pass

    def diff(self, n=1):
        """
        Take the difference between the current value and
        the nth value above it.

        Parameters
        ----------
        n: int

        Returns
        -------
        A DataFrame
        """

        def func():
            pass

        return self._non_agg(func)

    def pct_change(self, n=1):
        """
        Take the percentage difference between the current value and
        the nth value above it.

        Parameters
        ----------
        n: int

        Returns
        -------
        A DataFrame
        """

        def func():
            pass

        return self._non_agg(func)

    # ### Arithmetic and Comparison Operators ####

    def __add__(self, other):
        return self._oper("__add__", other)

    def __radd__(self, other):
        return self._oper("__radd__", other)

    def __sub__(self, other):
        return self._oper("__sub__", other)

    def __rsub__(self, other):
        return self._oper("__rsub__", other)

    def __mul__(self, other):
        return self._oper("__mul__", other)

    def __rmul__(self, other):
        return self._oper("__rmul__", other)

    def __truediv__(self, other):
        return self._oper("__truediv__", other)

    def __rtruediv__(self, other):
        return self._oper("__rtruediv__", other)

    def __floordiv__(self, other):
        return self._oper("__floordiv__", other)

    def __rfloordiv__(self, other):
        return self._oper("__rfloordiv__", other)

    def __pow__(self, other):
        return self._oper("__pow__", other)

    def __rpow__(self, other):
        return self._oper("__rpow__", other)

    def __gt__(self, other):
        return self._oper("__gt__", other)

    def __lt__(self, other):
        return self._oper("__lt__", other)

    def __ge__(self, other):
        return self._oper("__ge__", other)

    def __le__(self, other):
        return self._oper("__le__", other)

    def __ne__(self, other):
        return self._oper("__ne__", other)

    def __eq__(self, other):
        return self._oper("__eq__", other)

    def _oper(self, op, other):
        """
        Generic operator function

        Parameters
        ----------
        op: str name of special method
        other: the other object being operated on

        Returns
        -------
        A DataFrame
        """
        pass

    def sort_values(self, by, asc=True):
        """
        Sort the DataFrame by one or more values

        Parameters
        ----------
        by: str or list of column names
        asc: boolean of sorting order

        Returns
        -------
        A DataFrame
        """
        pass

    def sample(self, n=None, frac=None, replace=False, seed=None):
        """
        Randomly samples rows the DataFrame

        Parameters
        ----------
        n: int
            number of rows to return
        frac: float
            Proportion of the data to sample
        replace: bool
            Whether or not to sample with replacement
        seed: int
            Seeds the random number generator

        Returns
        -------
        A DataFrame
        """
        pass

    def pivot_table(self, rows=None, columns=None, values=None, aggfunc=None):
        """
        Creates a pivot table from one or two 'grouping' columns.

        Parameters
        ----------
        rows: str of column name to group by
            Optional
        columns: str of column name to group by
            Optional
        values: str of column name to aggregate
            Required
        aggfunc: str of aggregation function

        Returns
        -------
        A DataFrame
        """
        pass

    def _add_docs(self):
        agg_names = [
            "min",
            "max",
            "mean",
            "median",
            "sum",
            "var",
            "std",
            "any",
            "all",
            "argmax",
            "argmin",
        ]
        agg_doc = """
        Find the {} of each column

        Returns
        -------
        DataFrame
        """
        for name in agg_names:
            getattr(DataFrame, name).__doc__ = agg_doc.format(name)


class StringMethods:
    def __init__(self, df):
        self._df = df

    def capitalize(self, col):
        return self._str_method(str.capitalize, col)

    def center(self, col, width, fillchar=None):
        if fillchar is None:
            fillchar = " "
        return self._str_method(str.center, col, width, fillchar)

    def count(self, col, sub, start=None, stop=None):
        return self._str_method(str.count, col, sub, start, stop)

    def endswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.endswith, col, suffix, start, stop)

    def startswith(self, col, suffix, start=None, stop=None):
        return self._str_method(str.startswith, col, suffix, start, stop)

    def find(self, col, sub, start=None, stop=None):
        return self._str_method(str.find, col, sub, start, stop)

    def len(self, col):
        return self._str_method(str.__len__, col)

    def get(self, col, item):
        return self._str_method(str.__getitem__, col, item)

    def index(self, col, sub, start=None, stop=None):
        return self._str_method(str.index, col, sub, start, stop)

    def isalnum(self, col):
        return self._str_method(str.isalnum, col)

    def isalpha(self, col):
        return self._str_method(str.isalpha, col)

    def isdecimal(self, col):
        return self._str_method(str.isdecimal, col)

    def islower(self, col):
        return self._str_method(str.islower, col)

    def isnumeric(self, col):
        return self._str_method(str.isnumeric, col)

    def isspace(self, col):
        return self._str_method(str.isspace, col)

    def istitle(self, col):
        return self._str_method(str.istitle, col)

    def isupper(self, col):
        return self._str_method(str.isupper, col)

    def lstrip(self, col, chars):
        return self._str_method(str.lstrip, col, chars)

    def rstrip(self, col, chars):
        return self._str_method(str.rstrip, col, chars)

    def strip(self, col, chars):
        return self._str_method(str.strip, col, chars)

    def replace(self, col, old, new, count=None):
        if count is None:
            count = -1
        return self._str_method(str.replace, col, old, new, count)

    def swapcase(self, col):
        return self._str_method(str.swapcase, col)

    def title(self, col):
        return self._str_method(str.title, col)

    def lower(self, col):
        return self._str_method(str.lower, col)

    def upper(self, col):
        return self._str_method(str.upper, col)

    def zfill(self, col, width):
        return self._str_method(str.zfill, col, width)

    def encode(self, col, encoding="utf-8", errors="strict"):
        return self._str_method(str.encode, col, encoding, errors)

    def _str_method(self, method, col, *args):
        pass


def read_csv(fn):
    """
    Read in a comma-separated value file as a DataFrame

    Parameters
    ----------
    fn: string of file location

    Returns
    -------
    A DataFrame
    """
    pass
