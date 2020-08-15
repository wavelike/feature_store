import hashlib
import os
import pickle
import uuid

import pandas as pd


class FeatureStore:
    """
    This feature store uses a feature registry dictionary to map feature identifiers to filenames in order to make them retrievable later on.
    With the 'add_feature' method a pandas Series object is added to the feature registry and stored to disk.
    By providing a feature identifier to the method 'load_feature' the feature's values can be retrieved
    """

    def __init__(self,
                 storage_folderpath,
                 storage_name,
                 feature_folderpath,
                 ):

        self.storage_folderpath = storage_folderpath
        self.storage_name = storage_name
        self.feature_folderpath = feature_folderpath

        os.makedirs(feature_folderpath, exist_ok=True)

        self.feature_registry = {}

    @staticmethod
    def load_or_initialise(storage_folderpath: str,
                           storage_name: str,
                           feature_folderpath: str,
                           ):
        """
        Factory method:
        - Initialises a new feature store object if no file is found in the specified location
        - If the specified file exists, load the existing feature store from file.
        """

        filename = os.path.join(storage_folderpath, storage_name)
        if os.path.exists(filename):
            registry: FeatureStore = pickle.load(open(filename, 'rb'))
            return registry
        else:
            return FeatureStore(storage_folderpath, storage_name, feature_folderpath)


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

    @staticmethod
    def identifier_to_hash(identifier: str):
        # use hashlib library to create the hash, since its hashing method is python process independent.
        # The python internal __hash__ method relies on the PYTHONHASHSEED environment variable, which for security reasons
        # is set to a random value for each process
        return int(hashlib.md5(identifier.encode('utf-8')).hexdigest(), 16)

    def save_to_filesystem(self):
        filename = os.path.join(self.storage_folderpath, self.storage_name)
        pickle.dump(self, open(filename, 'wb+'))

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

    def get_filepath(self, identifier: str) -> str:

        identifier_string_hash = self.identifier_to_hash(identifier)
        stored_filename = self.feature_registry[identifier_string_hash]['filename']
        stored_filepath = os.path.join(self.feature_folderpath, stored_filename)

        return stored_filepath

