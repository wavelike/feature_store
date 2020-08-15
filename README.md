# Feature Store for faster Feature Engineering

***
**This article is taken from my github page: [wavelike.github.io](wavelike.github.io)**

To run the code, execute `code/m5_forecasting_example.py`
***

The process of feature engineering can be very repetitive and inefficient when being done manually and in an unstructured way.
In particular, ordinary feature engineering has the downsides that:
- the engineering process itself is not formalised, which leads to quick and dirty coding and try and error approaches
- it is hard to remember the computation methods that were already tried, leading to unneeded calculation repetitions
- calculated features are not preserved

In order to overcome these inefficiencies and speed up the feature engineering process, I came up with the following solution:

**Structuring of the engineering process**\
Each feature engineering method is part of an object that is initialised with calculation parameters. The class also provides a method to create feature identifier strings.

**Usage of a feature store and a feature registry**\
Each calculated feature is registered in the feature registry with its identifier and the feature data is stored efficiently to disk. Using the identifier string, it can easily be reloaded.



# Structuring of the engineering process
The main idea here is to embed the actual feature calculation method in a python class. Each object instantiated from this class is initialised with the parameters used for the actual feature calculation (e.g. rolling window size) and a raw data identifier. This identifier is obtained from the statistical properties of the source feature used for the engineering task.

The method's parameters together with the raw data identifier form the engineered feature identifier via the `create_identifier()` method. With this identifier the feature registry can determine whether an identical feature was previously calculated in order to load it from disk.

Each of these engineering processes inherits from the *FeatureProcess* class that acts like an interface each process needs to implement.


```python
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

```

With this interface we can define a feature process, e.g. one that calculates rolling window averages.

```python
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
```

Each process gets assigned a name that acts as the feature's column name in a pandas dataframe.


# Feature Store

The feature store is able to both remember and load previously calculated features.
For this purpose it uses a feature registry.

```python
class FeatureStore:

    ...
    ...

    def add_feature(self, identifier: str, feature: pd.Series):
        # Add new entry to the registry by mapping the feature identifier_string to the filepath

        unique_filename = f"{str(uuid.uuid1())}.parquet"
        feature_filepath = os.path.join(self.feature_folderpath, unique_filename)

        identifier_string_hash = self.identifier_to_hash(identifier)

        self.feature_registry.update(
            {
                identifier_string_hash: {
                    'filename': unique_filename,
                    'creation_timestamp': pd.Timestamp.now()
                }
            }
        )

        # Save the feature store object to the filesystem
        self.save_to_filesystem()

        # Store the feature to the filesystem
        pd.DataFrame(feature).to_parquet(feature_filepath)

    def registry_contains_identifier(self, identifier: str) -> bool:
        # Check if registry contains the provided identifier
        identifier_hash = self.identifier_to_hash(identifier)
        return identifier_hash in self.feature_registry.keys()


    def load_feature(self, identifier: str) -> pd.DataFrame:

        feature_filepath = self.get_filepath(identifier=identifier)

        print("Load from file...")

        feature = pd.read_parquet(feature_filepath)
        print("\t...loaded")

        return feature

```

When adding a feature to the store, a unique filename is created and an entry added to the registry that maps the identifier to the filename.
Then the feature store object itself is stored to the filesystem in order to make it usable after the current python process finished, and finally the feature's data is stored in parquet format.

The store can be loaded from disk in a later process and feature identifiers can be looked up via `contains_identifier_string()`. In case the identifier exists in the registry,
the feature can be directly loaded via `load_feature()` without the need of recalculating it.

# Feature engineering example and speed comparison
To demonstrate the usefulness of this approach, I use a rather large dataset from the Kaggle M5 Forecasting challenge ([link](https://www.kaggle.com/c/m5-forecasting-accuracy/data)) with several gigabytes of memory consumption, so the feature engineering is expected to be time consuming.

A few feature processes are defined and two run-throughs executed, one in which each feature is calculated, and one in which each feature is loaded from disk via the feature store.

The whole code to run this example can be found on my github: [link](https://github.com/wavelike/feature_store)

The resulting execution time for both approaches is shown in the bar chart.

![](time_comparison.png)

# Conclusion
Usage of the feature store accelerates the runtime significantly, which is not surprising since the features are loaded and not recomputed. Usage of such a simple feature store is a very helpful tool when doing extensive Feature Engineering, especially when done on a large dataset and over a longer time period.