import pandas as pd
import pytest 

from src.modeling.classification import split_data
from src.modeling.validation import check_duplicate_leakage

def create_test_dataset():
    """Create a small binary classificaiton dataset for testing."""

    return pd.DataFrame(
        {
            "BMI":[20,22,25,28,30,32,35,38,40,42],
            "Age":[2,3,4,5,6,7,8,9,10,11],
            "HighBP":[0,0,0,0,1,1,1,1,1,1],
            "Diabetes_binary":[0,0,0,0,0,1,1,1,1,1]
        }
    )
def test_split_data_returns_expected_components():
    data = create_test_dataset()
    result = split_data(data=data, target_column="Diabetes_binary", test_size=0.2, random_state=42, stratify=True)
    
    assert "X_train" in result
    assert "X_test" in result
    assert "y_train" in result
    assert "y_test" in result
    assert "metadata" in result

def test_split_data_preserved_row_count():
    data = create_test_dataset()
    result = split_data(data=data, target_column="Diabetes_binary")
    total_rows= (len(result["X_train"]) + len(result["X_test"]))
    assert total_rows == len(data)

def test_split_data_rejects_missing_target():
    data = create_test_dataset()
    with pytest.raises(Exception):
        split_data(data=data, target_column="MissingTarget")

def test_split_data_is_reproducible():
    data = create_test_dataset()
    result_1 = split_data(data=data, target_column="Diabetes_binary", random_state=42)
    result_2 = split_data(data=data, target_column="Diabetes_binary", random_state=42)
    pd.testing.assert_frame_equal(result_1['X_train'], result_2['X_train'])
    pd.testing.assert_series_equal(result_1['y_train'], result_2['y_train'])

def test_duplicate_leakage_check_passes():
    train = pd.DataFrame(
        {
            "BMI": [20, 25, 30],
            "Age": [2,4,6]
        }
    )

    test = pd.DataFrame(
        {
            "BMI": [35, 40],
            "Age": [0, 10]
        }
    )
    result = check_duplicate_leakage(
        X_train=train,
        X_test=test
    )
    assert result["leakage_detected"] is False
    assert result['status'] == "PASS"

def test_duplicate_leakage_check_detects_overlap():
    train = pd.DataFrame(
        {
            "BMI": [10, 25, 30],
            "Age": [2,4,6]
        }
    )

    test = pd.DataFrame(
        {
            "BMI": [30, 40],
            "Age": [6, 10]
        }
    )

    result = check_duplicate_leakage(
        X_train=train,
        X_test=test
    )

    assert result['leakage_detected'] is True
    assert result['status'] == "FAIL"
    assert result["overlapping_unique_rows"] == 1


def create_dataset_with_conflicting_duplicates():
    """10 feature vectors appear twice with opposite outcomes, plus 20 unique rows.

    unique_profile_v1 keeps such pairs because the complete rows differ.
    """
    paired = pd.DataFrame({
        "BMI": [20 + i for i in range(10)] * 2,
        "Age": [1 + i for i in range(10)] * 2,
        "Diabetes_binary": [0] * 10 + [1] * 10,
    })
    unique = pd.DataFrame({
        "BMI": [50 + i for i in range(20)],
        "Age": [1 + (i % 13) for i in range(20)],
        "Diabetes_binary": [0] * 10 + [1] * 10,
    })
    return pd.concat([paired, unique], ignore_index=True)


@pytest.mark.parametrize("stratify", [True, False])
def test_split_keeps_identical_feature_vectors_together(stratify):
    data = create_dataset_with_conflicting_duplicates()
    result = split_data(data=data, target_column="Diabetes_binary", stratify=stratify)

    train_vectors = set(map(tuple, result["X_train"].to_numpy()))
    test_vectors = set(map(tuple, result["X_test"].to_numpy()))
    assert train_vectors.isdisjoint(test_vectors)
    assert check_duplicate_leakage(result["X_train"], result["X_test"])["status"] == "PASS"


@pytest.mark.parametrize("seed", range(10))
def test_split_never_leaks_across_seeds(seed):
    data = create_dataset_with_conflicting_duplicates()
    result = split_data(data=data, target_column="Diabetes_binary", random_state=seed)
    assert result["metadata"]["leakage_check"]["leakage_detected"] is False


def test_split_metadata_records_grouping():
    data = create_dataset_with_conflicting_duplicates()
    metadata = split_data(data=data, target_column="Diabetes_binary")["metadata"]
    assert metadata["grouped_by_feature_vectors"] is True
    assert metadata["leakage_check"]["status"] == "PASS"
    assert metadata["actual_test_fraction"] == pytest.approx(0.2, abs=0.1)


def test_split_preserves_both_classes_in_test():
    data = create_dataset_with_conflicting_duplicates()
    result = split_data(data=data, target_column="Diabetes_binary", stratify=True)
    assert set(result["y_test"].unique()) == {0, 1}