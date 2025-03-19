#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
The MetadataWizard(pymdwizard) software was developed by the
U.S. Geological Survey Fort Collins Science Center.
See: https://github.com/usgs/fort-pymdwizard for current project source code
See: https://usgs.github.io/fort-pymdwizard/ for current user documentation
See: https://github.com/usgs/fort-pymdwizard/tree/master/examples
    for examples of use in other scripts

License:            Creative Commons Attribution 4.0 International (CC BY 4.0)
                    http://creativecommons.org/licenses/by/4.0/

PURPOSE
------------------------------------------------------------------------------
Module for reading data from various formats into a Pandas dataframe


SCRIPT DEPENDENCIES
------------------------------------------------------------------------------
    This script is part of the pymdwizard package and is not intented to be
    used independently.  All pymdwizard package requirements are needed.

    See imports section for external packages used in this script as well as
    inter-package dependencies


U.S. GEOLOGICAL SURVEY DISCLAIMER
------------------------------------------------------------------------------
This software has been approved for release by the U.S. Geological Survey
(USGS). Although the software has been subjected to rigorous review,
the USGS reserves the right to update the software as needed pursuant to
further analysis and review. No warranty, expressed or implied, is made by
the USGS or the U.S. Government as to the functionality of the software and
related material nor shall the fact of release constitute any such warranty.
Furthermore, the software is released on condition that neither the USGS nor
the U.S. Government shall be held liable for any damages resulting from
its authorized or unauthorized use.

Any use of trade, product or firm names is for descriptive purposes only and
does not imply endorsement by the U.S. Geological Survey.

Although this information product, for the most part, is in the public domain,
it also contains copyrighted material as noted in the text. Permission to
reproduce copyrighted items for other than personal use must be secured from
the copyright owner.
------------------------------------------------------------------------------
"""

from pathlib import Path
import struct
import datetime
import decimal

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip

try:
    xrange
except NameError:
    xrange = range

import pandas as pd
import numpy as np

try:
    import geopandas as gpd
    import fiona
except:
    gpd = None
    fiona = None

from pymdwizard.core import utils


def to_path_object(fname: str) -> Path:
    """
    Convert the file path to a Path object and check if the file exists.

    Parameters
    ----------
    fname : string
        full path to file

    Returns
    -------
    Path
        full path to file as a Path object.
    """

    try:
        fname = Path(fname)
        if not fname.exists():
            raise FileExistsError(f"Could not find file {fname}. Check path.")
        else:
            return fname
    except Exception as error:
        raise TypeError(f"Could not convert file name to a Path object. {error}")


def file_is_large(fname, large: int = 1e9) -> bool:
    """
    Check to see how large the file is to read in.

    Parameters
    ----------
    fname : string or pathlib.Path
            full path to file to read in

    large : int
            What is considered large. Default is 1E9 bytes (1Gb)

    Returns
    -------
    bool
        True if the file size is larger than `large`, False if not.
    """
    fname = to_path_object(fname)

    file_size = fname.stat().st_size

    if file_size / large > 1:
        return True
    return False


def get_file_encoding(fname: Path, delimiter: str = ",") -> str:
    """
    Get file encoding of a CSV through trial an error.  There
    maybe a better way to identify the encoding using chardet

    encoding_list = []
    with open(fn, "rb") as fid:
        for line in fid.readlines(5000):
            encoding_list.append(chardet.detect(line))
    encoding_df = pd.DataFrame(encoding_list)
    return(encoding_df["encoding"].mode()[0])

    Parameters
    ----------
    fname : Path
        full path to file
    delimiter : str, optional
        delimiter of file, by default ","

    Returns
    -------
    str
        encoding [None (ascii) | utf8 | ISO-8859-1]

    Raises
    ------
    UnicodeEncodeError
        _description_
    """
    n_rows = 10
    try:
        df = pd.read_csv(
            fname,
            parse_dates=True,
            delimiter=delimiter,
            nrows=n_rows,
            na_filter=False,
            comment="#",
        )
        return
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(
                fname,
                parse_dates=True,
                encoding="utf8",
                delimiter=delimiter,
                nrows=n_rows,
                na_filter=False,
                comment="#",
            )
            return "utf8"
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(
                    fname,
                    parse_dates=True,
                    encoding="ISO-8859-1",
                    delimiter=delimiter,
                    nrows=n_rows,
                    na_filter=False,
                    comment="#",
                )
                return "ISO-8859-1"
            except UnicodeDecodeError:
                raise UnicodeEncodeError(f"Could not decode {fname}")


def chunk_read(
    fname: Path, delimiter: str, encoding: str = None, chunk_size: int = 1e6
) -> pd.DataFrame:
    """
    Read a large CSV file in chunks and return just the min and max values
    for each column for the entire dataframe.

    Parameters
    ----------
    fname : Path
        full path to CSV file
    delimiter : str
        delimiter of CSV file
    encoding : str, optional
        encoding of the CSV file, default None
    chunk_size : int, optional
        chunk size, by default 1E6

    Returns
    -------
    pd.DataFrame
        a data frame of just the min and max values for each column.
    """
    pd_iterator = pd.read_csv(
        fname,
        parse_dates=True,
        encoding=encoding,
        delimiter=delimiter,
        na_filter=False,
        comment="#",
        chunksize=chunk_size,
        iterator=True,
        memory_map=True,
    )

    min_max_df_list = []

    for chunk_df in pd_iterator:
        min_max_df_list.append(chunk_df.agg(["min", "max"]))

    min_max_df = pd.concat(min_max_df_list, ignore_index=True)
    return min_max_df.agg(["min", "max"])


def read_csv(fname, delimiter=","):
    """
    converts a csv, specified by filename, into a pandas dataframe

    Parameters
    ----------
    fname : string
            Full fname to the csv to return
    delimiter : str, optional, defaults to comma
            the character used to delimit the data in a txt file

    Returns
    -------
    pandas dataframe
    """
    encoding = get_file_encoding(fname, delimiter=delimiter)

    if file_is_large(fname):
        return chunk_read(fname, delimiter, encoding)

    else:
        df = pd.read_csv(
            fname,
            parse_dates=True,
            encoding="utf8",
            delimiter=delimiter,
            na_filter=False,
            comment="#",
            memory_map=True,
        )

        return df.agg(["min", "max"])


def read_shp(fname):
    """
    Returns a pandas dataframe of the attribute in a shapefile's dbf
     specified as a file path/name

    Parameters
    ----------
    fname : str
            file path/name to the shapefile being returned

    Returns
    -------
        pandas dataframe
    """
    df = gpd.read_file(fname)
    c = fiona.open(fname)
    list(c.schema["properties"].keys())

    df = df[[c for c in df.columns if c != "geometry"]]
    df.insert(0, "Shape", c.schema["geometry"])
    df.insert(0, "FID", range(df.shape[0]))
    return df.agg(["min", "max"])


def dbfreader(f):
    """Returns an iterator over records in a Xbase DBF file.
    The first row returned contains the field names.
    The second row contains field specs: (type, size, decimal places).
    Subsequent rows contain the data records.
    If a record is marked as deleted, it is skipped.
    File should be opened for binary reads.

    originally taken from:
        http://code.activestate.com/recipes/362715-dbf-reader-and-writer/

    See DBF format spec at:
        http://www.pgts.com.au/download/public/xbase.htm#DBF_STRUCT
    """

    numrec, lenheader = struct.unpack("<xxxxLH22x", f.read(32))
    numfields = (lenheader - 33) // 32

    fields = []
    for fieldno in xrange(numfields):
        name, typ, size, deci = struct.unpack("<11sc4xBB14x", f.read(32))
        name = bytes(name)
        name = name.replace(b"\0", b"")  #  eliminate NULs from string
        fields.append((name, typ, size, deci))
    yield [field[0] for field in fields]
    yield [tuple(field[1:]) for field in fields]

    terminator = f.read(1)
    assert terminator == b"\r"

    fields.insert(0, ("DeletionFlag", "C", 1, 0))
    fmt = "".join(["%ds" % fieldinfo[2] for fieldinfo in fields])
    fmtsiz = struct.calcsize(fmt)
    for i in xrange(numrec):
        record = struct.unpack(fmt, f.read(fmtsiz))
        if record[0] != b" ":
            continue  #  deleted record
        result = []
        for (name, typ, size, deci), value in izip(fields, record):
            value = bytes(value)
            if name == "DeletionFlag":
                continue
            if typ == b"N":
                value = value.replace(b"\0", b"").lstrip()
                if value == "":
                    value = 0
                elif deci:
                    value = decimal.Decimal(value)
                else:
                    value = int(value)
            if typ == b"C":
                value = value.decode("utf-8")
            elif typ == b"D":
                y, m, d = int(value[:4]), int(value[4:6]), int(value[6:8])
                value = datetime.date(y, m, d)
            elif typ == b"L":
                value = (
                    (value in b"YyTt" and b"T") or (value in b"NnFf" and b"F") or b"?"
                )
            elif typ == b"F":
                value = float(value)
            result.append(value)
        yield result


def read_dbf(fname):
    """
    Returns a pandas dataframe of the dbf
     specified as a file path/name

    Parameters
    ----------
    fname : str
            file path/name to the dbf being returned

    Returns
    -------
        pandas dataframe
    """
    f = open(fname, "rb")
    vat = list(dbfreader(f))
    return pd.DataFrame(vat[2:], columns=[c.decode("utf-8") for c in vat[0]])


def get_sheet_names(fname):
    """
    Returns list of sheets in an Excel file

    Parameters
    ----------
    fname : str
            file path/name to the Excel file being returned

    Returns
    -------
    list of strings
    """
    workbook = pd.ExcelFile(fname)
    return workbook.sheet_names


def read_excel(fname, sheet_name):
    """
    Returns a pandas dataframe of an Excel file and sheet

    Parameters
    ----------
    fname : str
            file path/name to the Excel file
    sheet_name : str

    Returns
    -------
        pandas dataframe
    """
    if fname.endswith(".xlsx") or fname.endswith(".xlsm"):
        df = pd.read_excel(fname, sheet_name, engine="openpyxl")
    else:
        df = pd.read_excel(fname, sheet_name)
    return df.agg(["min", "max"])


def read_las(fname):
    """
    Returns a pandas dataframe of the attribute in a las file

    Parameters
    ----------
    fname : str
            file path/name to the las file being returned

    Returns
    -------
        pandas dataframe
    """
    import laspy

    max_rows = int(utils.get_setting("maxrows", 1000000))

    las = laspy.open(fname)
    dims = [dim.name for dim in las.header.point_format]

    for points in las.chunk_iterator(max_rows):
        break

    point_data = {dim: np.array(points[dim]) for dim in dims}
    df = pd.DataFrame(point_data)

    return df.agg(["min", "max"])


def read_data(fname, sheet_name="", delimiter=","):
    """
    Returns pandas dataframe from a file (csv, txt, Excel, or shp)

    Parameters
    ----------
    fname : str
            file path/name to the Excel file
    sheet_name : str, optional
            sheet name
    delimiter : str, optional
            the character used to delimit the data in a txt file

    Returns
    -------
        pandas dataframe
    """
    if fname.lower().endswith(".csv"):
        return read_csv(fname)
    elif fname.lower().endswith(".txt"):
        return read_csv(fname, delimiter)
    elif fname.lower().endswith(".shp"):
        return read_shp(fname)
    elif fname.lower().endswith(".las") or fname.lower().endswith(".laz"):
        return read_las(fname)
    elif sheet_name:
        return read_excel(fname, sheet_name)


def sniff_nodata(series):
    """
    Attempt to guess the nodata value associated with a series

    Parameters
    ----------
    series : pandas series

    Returns
    -------
    str : the nodata placeholder in a series
    """
    uniques = series.uniques()

    for nd in [
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "n/a",
        "nan",
        "null",
        -9999,
        "-9999",
        "",
        "Nan",
    ]:
        if nd in list(uniques):
            return nd

    return None


def clean_nodata(series, nodata=None):
    """
    Given a series remove the values that match the specified nodata value
    and convert it to an int or float if possible

    Parameters
    ----------
    series : pandas series
    nodata : string, int, or float Nodata placeholder

    Returns
    -------
    pandas series
    """
    if nodata is None:
        return series

    clean_series = series[series != nodata]

    try:
        clean_series = clean_series.astype("int64")
    except ValueError:
        try:
            clean_series = clean_series.astype("float64")
        except ValueError:
            pass

    return clean_series


def get_df_info(df: pd.DataFrame, int_is_unique: bool = True) -> dict:
    """
    Get the information from a data frame that would be need to populate
    metadata.

    - If the column is of type float, then get the min/max values
    - If type int get unique values if int_is_unique = True, otherwise
      get min/max of ints
    - If string or object get unique values
    - If datetime get min/max

    Parameters
    ----------
    df : _type_
        _description_
    """
    described = df.describe(include="all")

    info_dict = {}

    def fill_numeric_dict(col, dataframe):
        info_dict[col] = {
            "unique_values": None,
            "min": dataframe.loc["min", col],
            "max": dataframe.loc["max", col],
            "std": dataframe.loc["std", col],
        }

    def fill_category_dict(col, dataframe):
        info_dict[col] = {
            "unique_values": dataframe[col].unique().to_list(),
            "min": None,
            "max": None,
            "std": None,
        }

    for col in df.columns:
        # if a string get unique values
        if df.dtypes[col].name in ["object"]:
            fill_category_dict(col, df)
        elif "int" in df.dtypes[col].name and int_is_unique == True:
            fill_category_dict(col, df)
        elif "int" in df.dtypes[col].name and int_is_unique == False:
            fill_numeric_dict(col, described)
        elif "datetime" in df.dtypes[col].name:
            fill_numeric_dict(col, described)
        else:
            fill_numeric_dict(col, described)

    return info_dict
