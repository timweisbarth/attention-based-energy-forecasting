from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import holidays


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5

# Make hourofday, dayofweek and dayofyear cyclic
# Need two dimensions (sin and cos) because only one would be ambgious (e.g. sin has y=0.5 at two locations)
# period of sin(bx) or cos(bx) is p = 2*pi / b. We want p to be period of 24 (hour), 7 (daysofweek) or 366(year incl. leap year)
class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        hours = index.hour.to_numpy()
        hour_sin = np.sin(2 * np.pi * hours / 24).reshape(1,-1)
        hour_cos = np.cos(2 * np.pi * hours / 24).reshape(1,-1)
        hours = np.vstack((hour_sin, hour_cos))
        return hours


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        dayofweek = index.dayofweek.to_numpy()
        dayofweek_sin = np.sin(2 * np.pi * dayofweek / 7).reshape(1,-1)
        dayofweek_cos = np.cos(2 * np.pi * dayofweek / 7).reshape(1,-1)
        dayofweek = np.vstack((dayofweek_sin, dayofweek_cos))
        return dayofweek


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        out = ((index.day - 1) / 30.0 - 0.5).to_numpy()
        #print("Month", type(out), out.shape)
        return out


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        dayofyear = index.dayofyear.to_numpy()
        #print(dayofyear.min(),dayofyear.max())
        dayofyear_sin = np.sin(2 * np.pi * dayofyear / 366).reshape(1,-1)
        dayofyear_cos = np.cos(2 * np.pi * dayofyear / 366).reshape(1,-1)
        dayofyear = np.vstack((dayofyear_sin, dayofyear_cos))
        return dayofyear


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        out = (index.month - 1) / 11.0 - 0.5
        #print("Month", type(out), out.shape)
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5
    
############## Added classes ##################

class Year(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        out = ((index.year - 2015) / 8.0 - 0.5).to_numpy().reshape(1, -1)
        #print("Year", type(out), out.shape, out.min(), out.max())
        return out
class IsHoliday(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        years = range(2015,2025,1)
        holidays_GER = [holiday for holiday in holidays.Germany(years=years)]
        df_dates = pd.DataFrame(index.date)
        out = (df_dates.isin(holidays_GER).values.astype(int) - 0.5).reshape(1, -1)
        #print("Holiday", type(out), out.shape, out.min(), out.max())
        return out

class IsWeekend(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        df_dayofweek = pd.DataFrame(index.dayofweek)
        is_saturday = df_dayofweek.isin([5]).values.astype(int)
        is_sunday = df_dayofweek.isin([6]).values.astype(int)
        out = (is_saturday + is_sunday - 0.5).reshape(1, -1)
        #print("Weekend", type(out), out.shape, out.min(), out.max())
        return out



def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfYear, IsHoliday, IsWeekend, Year], # I removed DayOfMonth
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }
    # for freq_str = "h" it returns <Hour>
    # for freq_str = "5min" it returns <5 * Minutes>
    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    out = np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
    return out
