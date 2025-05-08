import wandb
import pytest
import pandas as pd
import os

# Initialize a W&B run with separate entity and project
print("Initializing W&B...")
try:
    run = wandb.init(
        entity="ngocnhi-p4work-national-economics-university",  
        project="diabetes",  
        job_type="data_checks"
    )
    print("W&B initialization successful")
except Exception as e:
    print(f"Error during W&B initialization: {e}")
    raise

@pytest.fixture(scope="session")
def data():
    """
    Fixture to load the diabetes dataset from a W&B artifact.
    """
    print("Loading artifact...")
    try:
        artifact = run.use_artifact("preprocessed_data.csv:latest", type="clean_data")
        artifact_dir = artifact.download()  # Load the directory containing the artifact
        local_path = os.path.join(artifact_dir, "preprocessed_data.csv")  # Concatenate the path to the actual file
        print(f"Artifact has been downloaded at {local_path}")
        df = pd.read_csv(local_path)
        print(f"Data has been loaded successfully: {df.shape}")
        print("Columns:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Error while loading artifact or reading data: {e}")
        raise

def test_no_missing_values(data):
    assert data.isnull().sum().sum() == 0, "Data still contains missing values"

def test_class_balance(data, threshold=0.9):
    class_counts = data['OUTCOME'].value_counts(normalize=True)
    max_class_ratio = class_counts.max()
    assert max_class_ratio < threshold, f"Data is imbalanced: {class_counts.to_dict()}"

def test_duplicate_rows(data):
    duplicate_count = data.duplicated().sum()
    assert duplicate_count == 0, f"Data has {duplicate_count} duplicated rows"

def test_data_length(data):
    """
    Check if the dataset has enough rows to proceed.
    """
    print("Running test_data_length...")
    assert len(data) > 500, f"Dataset has {len(data)} rows, need > 500"

def test_number_of_columns(data):
    """
    Check if the dataset has the expected number of columns.
    """
    print("Running test_number_of_columns...")
    expected_min_columns = 9
    assert data.shape[1] >= expected_min_columns, f"Dataset has {data.shape[1]} columns, need >= {expected_min_columns} columns"


def test_column_presence_and_type(data):
    """
    Check data types: OUTCOME is an integer, the remaining columns are floats.
    """
    print("Running test_column_presence_and_type...")

    # 1. Ensure that the OUTCOME column is present
    assert "OUTCOME" in data.columns, "Missing column OUTCOME"

    # 2. Check the data type of OUTCOME
    assert pd.api.types.is_integer_dtype(data["OUTCOME"]), "The OUTCOME column is not of integer type"

    # 3. Check that the remaining columns are numeric
    feature_cols = [col for col in data.columns if col != "OUTCOME"]
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(data[col]), f"Column {col} is not in numeric format"


def test_class_names(data):
    """
    Check that the Outcome column contains only valid values (0 or 1).
    """
    print("Running test_class_names...")
    known_classes = [0, 1]
    assert data["OUTCOME"].isin(known_classes).all(), \
        f"The Outcome column contains invalid values: {data['OUTCOME'].unique()}"

def test_column_ranges(data):
    """
    Check that the numeric columns have values within a reasonable range (non-negative and not extreme).
    """
    print("Running test_column_ranges...")

    # Drop OUTCOME
    feature_cols = [col for col in data.columns if col != "OUTCOME" and pd.api.types.is_numeric_dtype(data[col])]

    for col in feature_cols:
        min_val, max_val = data[col].min(), data[col].max()

        # If all values are negative or very high, it could be an error
        assert min_val >= 0, f"Column {col} has negative value (min={min_val})"
        assert max_val < 1e6, f"Column {col} has very high value (max={max_val})"
