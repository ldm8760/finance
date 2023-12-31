from datetime import datetime
import numpy as np
from dateutil import parser
from datetime import timezone

def fmt_dt(date_input: str | datetime | int, form: str = None) -> str | datetime:
    if type(date_input) == str:
        result = parser.parse(date_input)
        if form is not None:
            result = getattr(result, form)()
    elif type(date_input) == datetime:
        result = date_input.strftime("%Y-%m-%d %H:%M:%S")
        if form is not None:
            result = getattr(date_input, form)()
    else:
        result = datetime.utcfromtimestamp(date_input)
    
    return result


def dt_parse(date_time_str: str) -> str:
    """Returns a date object"""

    try:
        ret = datetime.strftime(datetime.strptime(date_time_str, "%Y-%m-%d %H:%M:%S%z"), "%Y-%m-%d")
    except ValueError:
        ret = datetime.strftime(datetime.strptime(date_time_str[:-3], "%b %d, %Y, %I %p"), "%Y-%m-%d")
    return ret

def date_parse(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")

def remove_tz(date_time_tz: datetime) -> str:
    return datetime.strftime(datetime.strptime(str(date_time_tz), '%Y-%m-%d %H:%M:%S%z'), '%Y-%m-%d %H:%M:%S')

def to_datetime_type(datetime_str: str) -> datetime:
    return datetime.strptime(str(datetime_str), '%Y-%m-%d %H:%M:%S')

def to_datetype(datetime_str: str) -> datetime.date:
    return datetime.strptime(str(datetime_str)[:10], '%Y-%m-%d')


def two_dec(num: float) -> float:
    """Returns a two digit float of num"""
    return round(num, 2)


def y_len(iterable: any) -> any:
    """Shorthand form of range(len(obj))"""
    return np.arange(np.size(iterable))


def fix_data(data, datatype):
    data = data.replace(np.nan, 0)
    data = data.astype(datatype)
    return data
