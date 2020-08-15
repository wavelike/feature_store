from abc import abstractmethod
from typing import Union, Optional

import pandas as pd
import json
import numpy as np

class FeatureProcess:

    name: str
    identifier: str

    @abstractmethod
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        # Execution of the feature engineering method
        pass

    @abstractmethod
    def create_identifier(self) -> str:
        # Combines the process attributes (data identifier and process parameters) into a string that identifies the feature
        pass


class RollingAverageProcess(FeatureProcess):
    """
    Calculates the rolling window average
    """

    def __init__(self,
                 source_feature_name: str,
                 window: int,
                 source_identifier: str,
                 ):
        super().__init__()

        self.source_feature_name = source_feature_name
        self.source_identifier = source_identifier
        self.window = window

        self.name = f'rolling_avg_w{window}_{source_feature_name}'

        self.identifier = self.create_identifier()

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:

        data.loc[:, self.name] = data[self.source_feature_name].rolling(self.window).mean()

        return data

    def create_identifier(self) -> str:

        identifier_dict = {
            'name': self.name,
            'inputs': [
                {
                    'source_feature_name': self.source_feature_name,
                    'source_identifier': self.source_identifier,
                },
            ],
            'parameters': [
                {
                    'param_name': 'window',
                    'param_value': self.window
                }
            ],
        }

        identifier = json.dumps(identifier_dict, sort_keys=True)

        return identifier


class GroupAverageProcess(FeatureProcess):

    def __init__(self,
                 feature_to_group_by_name: str,
                 feature_to_group_by_identifier: str,
                 feature_to_average_name: str,
                 feature_to_average_identifier: str,
                 ):

        self.feature_to_group_by_name = feature_to_group_by_name
        self.feature_to_group_by_identifier = feature_to_group_by_identifier
        self.feature_to_average_name = feature_to_average_name
        self.feature_to_average_identifier = feature_to_average_identifier

        self.name = f'group_avg_{feature_to_group_by_name}_{feature_to_average_name}'

        self.identifier = self.create_identifier()

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:

        group_average = (
            data[[self.feature_to_group_by_name, self.feature_to_average_name]]
                .groupby(self.feature_to_group_by_name)
                .mean()
                .rename(columns={self.feature_to_average_name: self.name}))

        data = data.join(group_average, how='left', on=self.feature_to_group_by_name)

        return data

    def create_identifier(self):

        identifier_dict = {
            'label': self.name,
            'inputs': [
                {
                    'feature_to_group_by_name': self.feature_to_group_by_name,
                    'feature_to_group_by_identifier': self.feature_to_group_by_identifier,
                    'feature_to_average_name': self.feature_to_average_name,
                    'feature_to_average_identifier': self.feature_to_average_identifier,
                },
            ],
            'parameters': [],
        }

        identifier = json.dumps(identifier_dict, sort_keys=True)

        return identifier


class DailyRollingTrendGroupedProcess(FeatureProcess):

    def __init__(self,
                 feature_to_group_by_name: str,
                 feature_to_group_by_identifier: str,
                 date_feature_name: str,
                 date_feature_name_identifier: str,
                 trend_feature_name: str,
                 trend_feature_name_identifier: str,
                 window: int,
                 ):
        """

        :param feature_to_group_by_name: Raw feature name by which to group the data
        :param feature_to_group_by_identifier:
        :param date_feature_name: Raw feature name that contains the date information
        :param date_feature_name_identifier:
        :param trend_feature_name: Raw feature name for which the recent linear trend is calculated
        :param trend_feature_name_identifier:
        :param window: Rolling window length
        """

        self.feature_to_group_by_name = feature_to_group_by_name
        self.feature_to_group_by_identifier = feature_to_group_by_identifier
        self.date_feature_name = date_feature_name
        self.date_feature_name_identifier = date_feature_name_identifier
        self.trend_feature_name = trend_feature_name
        self.trend_feature_name_identifier = trend_feature_name_identifier

        self.window = window

        self.name = f'daily_rolling_w{window}_trend_grouped_{feature_to_group_by_name}_{date_feature_name}_{trend_feature_name}'

        self.identifier = self.create_identifier()

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:

        # First Group the data by date and get the mean of the trend feature
        average_daily_values_by_group = (
            data[[self.feature_to_group_by_name, self.date_feature_name, self.trend_feature_name]]
                .groupby([self.feature_to_group_by_name, self.date_feature_name])
                .mean()
                .reset_index())

        # Group the reduced data again
        groups = average_daily_values_by_group[[self.feature_to_group_by_name, self.date_feature_name, self.trend_feature_name]].groupby(self.feature_to_group_by_name)

        # Iterate over each group, apply a rolling window over previous dates, apply a linear regression and retrieve the slope (=trend)
        trend_data = pd.DataFrame()
        for group_index, group_data in groups:
            group_data.loc[:, self.name] = group_data.rolling(self.window, min_periods=2)[self.trend_feature_name].apply(self.get_recent_trend, raw=True)
            trend_data = trend_data.append(group_data)

        # merge the trend data on the original data by date and group value
        data = data.merge(trend_data[[self.feature_to_group_by_name, self.date_feature_name, self.name]], how='left', on=[self.feature_to_group_by_name, self.date_feature_name])

        return data

    @staticmethod
    def get_recent_trend(data: np.ndarray) -> Optional[float]:
        # Calculates the linear slope of the provided data

        if len(data) >= 2:
            linear_fit = np.polyfit(range(len(data)), data, 1)
            trend = linear_fit[0]
        else:
            trend = np.nan

        return trend

    def create_identifier(self):
        identifier_dict = {
            'name': self.name,
            'inputs': [
                {
                    'source_feature_name_to_group': self.feature_to_group_by_name,
                    'source_feature_name_date': self.date_feature_name,
                    'source_feature_name_for_trend': self.trend_feature_name,
                    'data_identifier_feature_to_group': self.feature_to_group_by_identifier,
                    'data_identifier_feature_to_average': self.date_feature_name_identifier,
                    'data_identifier_feature_for_trend': self.trend_feature_name_identifier,
                },
            ],
            'parameters': [
                {
                    'param_name': 'window',
                    'param_value': self.window
                }
            ],
        }

        identifier = json.dumps(identifier_dict, sort_keys=True)

        return identifier