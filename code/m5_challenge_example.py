from typing import List, Tuple

from pathlib import Path
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns


from code.feature_processes import RollingAverageProcess, GroupAverageProcess, DailyRollingTrendGroupedProcess, FeatureProcess
from code.data_identifier import calc_data_identifier
from code.data_preparation import get_data
from code.feature_store import FeatureStore


def prepare_dataset(keep_day_ratio, n_items_ids):

    dataset_name = f'data_sample_{keep_day_ratio}_{n_items_ids}'
    dataset_filepath = Path(os.path.join(os.getcwd(), 'data', dataset_name + '.parquet'))

    dataset_filepath.parent.mkdir(parents=True, exist_ok=True)
    if dataset_filepath.exists():
        print("File already exists.")
        data = pd.read_parquet(dataset_filepath)
    else:
        print("Building dataset file...")
        data = get_data(keep_day_ratio=keep_day_ratio,
                        n_item_ids=n_items_ids,
                        )
        data.to_parquet(dataset_filepath)

    return data


def get_feature_store(storage_name):

    storage_folderpath = Path(os.path.join(os.getcwd(), 'feature_store'))
    storage_folderpath.mkdir(parents=True, exist_ok=True)
    feature_folderpath = Path(os.path.join(str(storage_folderpath), 'features'))
    feature_folderpath.parent.mkdir(parents=True, exist_ok=True)

    feature_store = FeatureStore.load_or_initialise(storage_folderpath=storage_folderpath,
                                                    storage_name=storage_name,
                                                    feature_folderpath=feature_folderpath,
                                                    )

    return feature_store


def execute_feature_engineering(data: pd.DataFrame,
                                feature_store: FeatureStore,
                                feature_processes: List[FeatureProcess],
                                force_feature_calculation: bool
                                ) -> Tuple[pd.DataFrame, pd.Series]:
    # Iterates over each feature process and either calculates the feature and adds it to the feature store, or loads the feature from the feature store

    time_by_feature = pd.Series()
    time_by_feature.name = 'time'
    for feature_process in feature_processes:

        print(f"Feature Process '{feature_process.name}'")

        start_time = time.time()

        feature_identifier = feature_process.create_identifier()

        if (not feature_store.registry_contains_identifier(feature_identifier) or force_feature_calculation):

            data = feature_process.execute(data=data)
            feature_store.add_feature(identifier=feature_identifier, feature=data[feature_process.name])
            print(f"\tCalculated and added to feature store")

        else:

            try:
                feature = feature_store.load_feature(identifier=feature_identifier)
                data[feature_process.name] = feature.values
                print("\tLoaded from file ")
            except Exception as e:
                print(f"\tNot loadable via feature store")
                raise e

        end_time = time.time()
        time_diff = end_time - start_time
        time_by_feature[feature_process.name] = time_diff

        print(f"\tDone feature {feature_process.name}, time: {time_diff}")

    return data, time_by_feature




if __name__ == '__main__':

    time_by_feature = pd.DataFrame()

    data = prepare_dataset(keep_day_ratio=1/50, # None # 50
                           n_items_ids=500,  # 500
                           )
    raw_features = data.columns
    feature_store = get_feature_store(storage_name='feature_store.pkl')

    # Calculate data identifiers for all raw data features used in the feature engineering processes
    source_raw_features = [
        'sales',
        'date',
        'item_id',
        'store_id',
    ]

    data_identifiers = {}
    for col in source_raw_features:
        data_identifiers[col] = calc_data_identifier(feature=data[col])

    # Define all Feature Processes
    feature_processes = [
        RollingAverageProcess(source_feature_name='sales',
                              window=1000,
                              source_identifier=data_identifiers['sales'],),

        GroupAverageProcess(feature_to_group_by_name='item_id',
                            feature_to_average_name='sales',
                            feature_to_group_by_identifier=data_identifiers['item_id'],
                            feature_to_average_identifier=data_identifiers['sales'],),

        GroupAverageProcess(feature_to_group_by_name='store_id',
                            feature_to_average_name='sales',
                            feature_to_group_by_identifier=data_identifiers['item_id'],
                            feature_to_average_identifier=data_identifiers['sales'], ),

        DailyRollingTrendGroupedProcess(feature_to_group_by_name='store_id',
                                        date_feature_name='date',
                                        trend_feature_name='sales',
                                        window=30,
                                        feature_to_group_by_identifier=data_identifiers['store_id'],
                                        date_feature_name_identifier=data_identifiers['date'],
                                        trend_feature_name_identifier=data_identifiers['sales'],),
    ]

    # Start the feature calculation processes
    for method in ['calculate', 'load']:
        # Execute the feature engineering processes - Either force calculation or allow loading the features from the feature store

        if method == 'calculate':
            data_engineered, time_by_feature_calculated = execute_feature_engineering(data[raw_features],
                                                                                      feature_store=feature_store,
                                                                                      feature_processes=feature_processes,
                                                                                      force_feature_calculation=True)
            time_by_feature_calculated = pd.DataFrame(time_by_feature_calculated)
            time_by_feature_calculated['method'] = 'calculated'
            time_by_feature = time_by_feature.append(time_by_feature_calculated)

        if method == 'load':
            data_engineered_loaded, time_by_feature_loaded = execute_feature_engineering(data[raw_features],
                                                                                         feature_store=feature_store,
                                                                                         feature_processes=feature_processes,
                                                                                         force_feature_calculation=False)
            time_by_feature_loaded = pd.DataFrame(time_by_feature_loaded)
            time_by_feature_loaded['method'] = 'loaded'
            time_by_feature = time_by_feature.append(time_by_feature_loaded)

    # Compare execution times of methods 'calculated' and 'loaded'
    time_by_feature = time_by_feature.reset_index().rename(columns={'index': 'feature_label', 0: 'time'})

    time_by_feature['data_length'] = len(data)

    # Plot method comparison for each feature
    unique_feature_labels = time_by_feature['feature_label'].unique()
    for unique_label in unique_feature_labels:
        ax = sns.barplot(x="feature_label", y="time", hue="method", data=time_by_feature[time_by_feature['feature_label'] == unique_label])
        plt.show()

    # Plot method comparison overall
    time_by_method = time_by_feature[['method', 'time']].groupby('method').sum()
    time_by_method = time_by_method.sort_values('time', ascending=False)

    ax = time_by_method.plot.bar(title=f"Execution times: Calculating vs. loading features")

    plt.ylabel('time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'time_comparison.png'))

    plt.show()