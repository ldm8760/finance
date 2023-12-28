from datetime import datetime
import numpy as np
import re

date_formats = {
        'year': '%Y',
        'month': '%m',
        'day': '%d',
        'hour': '%H',
        'minute': '%M',
        'second': '%S',
}
additional_formats = {
    'month': '%b',
    'day': '%d',
    'year': '%Y',
    'hour': '%I',
    'minute': '%M',
    'period': '%p',
}


def get_date_formats(date_str: str) -> datetime:
    """Automatically generates a dictionary of date components and format strings"""

    result_list = re.split(r'(-|:|,| )', date_str)
    dt_ints = [item for item in result_list if item.isnumeric()]
    dict_formats = date_formats.copy()

    for i in range(len(dt_ints)):
        for format_name, format_str in dict_formats.items():
            try:
                if format_name == 'second' and '00' in result_list:
                    result_list = [format_str if value == '00' else value for value in result_list]

                parse_result = datetime.strptime(dt_ints[i], format_str)
                parsed_value = str(getattr(parse_result, format_name))

                result_list = [format_str if value == parsed_value else value for value in result_list]

                
                del dict_formats[format_name]
                break
            except ValueError:
                print(f'{dt_ints[i]} does not match {format_name} format')
            except Exception as e:
                print(f'{e}')

    back_str = ''.join(result_list)

    return datetime.strptime(date_str, back_str)


# Example usage:
input_date = "2023-12-27, 15:30:00"
# input_date = "Dec 27, 2023, 03 PM"
auto_generated_formats = get_date_formats(input_date)
print(auto_generated_formats)



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
