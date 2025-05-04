import wandb
import pytest
import pandas as pd
import os

# Khởi tạo W&B run với entity và project tách biệt
print("Đang khởi tạo W&B...")
try:
    run = wandb.init(
        entity="ngocnhi-p4work-national-economics-university",  # Tên entity
        project="diabetes",  # Tên project
        job_type="data_checks"
    )
    print("Khởi tạo W&B thành công")
except Exception as e:
    print(f"Lỗi khi khởi tạo W&B: {e}")
    raise

@pytest.fixture(scope="session")
def data():
    """
    Fixture để tải tập dữ liệu diabetes từ W&B artifact.
    """
    print("Đang tải artifact...")
    try:
        artifact = run.use_artifact("preprocessed_data.csv:latest", type="clean_data")
        artifact_dir = artifact.download()  # Tải thư mục chứa artifact
        local_path = os.path.join(artifact_dir, "preprocessed_data.csv")  # Nối đường dẫn đến file thực tế
        print(f"Artifact đã được tải về tại {local_path}")
        df = pd.read_csv(local_path)
        print(f"Dữ liệu đã được tải thành công: {df.shape}")
        print("Các cột:", df.columns.tolist())
        return df
    except Exception as e:
        print(f"Lỗi khi tải artifact hoặc đọc dữ liệu: {e}")
        raise

def test_no_missing_values(data):
    assert data.isnull().sum().sum() == 0, "Dữ liệu còn chứa missing values"

def test_class_balance(data, threshold=0.9):
    class_counts = data['OUTCOME'].value_counts(normalize=True)
    max_class_ratio = class_counts.max()
    assert max_class_ratio < threshold, f"Dữ liệu mất cân bằng: {class_counts.to_dict()}"

def test_duplicate_rows(data):
    duplicate_count = data.duplicated().sum()
    assert duplicate_count == 0, f"Dữ liệu có {duplicate_count} dòng trùng lặp"

def test_data_length(data):
    """
    Kiểm tra xem tập dữ liệu có đủ số hàng để tiếp tục không.
    """
    print("Đang chạy test_data_length...")
    assert len(data) > 500, f"Tập dữ liệu có {len(data)} hàng, cần > 500"

def test_number_of_columns(data):
    """
    Kiểm tra xem tập dữ liệu có đúng số cột mong đợi không.
    """
    print("Đang chạy test_number_of_columns...")
    expected_min_columns = 9
    assert data.shape[1] >= expected_min_columns, f"Tập dữ liệu có {data.shape[1]} cột, cần >= {expected_min_columns} cột"


def test_column_presence_and_type(data):
    """
    Kiểm tra kiểu dữ liệu: OUTCOME là số nguyên, các cột còn lại là số thực.
    """
    print("Đang chạy test_column_presence_and_type...")

    # 1. Đảm bảo có cột OUTCOME
    assert "OUTCOME" in data.columns, "Thiếu cột OUTCOME"

    # 2. Kiểm tra kiểu dữ liệu của OUTCOME
    assert pd.api.types.is_integer_dtype(data["OUTCOME"]), "Cột OUTCOME không phải kiểu số nguyên"

    # 3. Kiểm tra các cột còn lại là numeric
    feature_cols = [col for col in data.columns if col != "OUTCOME"]
    for col in feature_cols:
        assert pd.api.types.is_numeric_dtype(data[col]), f"Cột {col} không phải dạng số (numeric)"


def test_class_names(data):
    """
    Kiểm tra cột Outcome chỉ chứa các giá trị hợp lệ (0 hoặc 1).
    """
    print("Đang chạy test_class_names...")
    known_classes = [0, 1]
    assert data["OUTCOME"].isin(known_classes).all(), \
        f"Cột Outcome chứa giá trị không hợp lệ: {data['OUTCOME'].unique()}"

def test_column_ranges(data):
    """
    Kiểm tra các cột số có giá trị nằm trong khoảng hợp lý (không âm và không cực đoan).
    """
    print("Đang chạy test_column_ranges...")

    # Bỏ OUTCOME
    feature_cols = [col for col in data.columns if col != "OUTCOME" and pd.api.types.is_numeric_dtype(data[col])]

    for col in feature_cols:
        min_val, max_val = data[col].min(), data[col].max()

        # Nếu tất cả giá trị đều âm hoặc rất cao, có thể là lỗi
        assert min_val >= 0, f"Cột {col} chứa giá trị âm (min={min_val})"
        assert max_val < 1e6, f"Cột {col} có giá trị cực lớn (max={max_val})"
