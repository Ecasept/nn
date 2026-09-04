#!/bin/sh

set -eu

dataset_url="https://www.kaggle.com/api/v1/datasets/download/oddrationale/mnist-in-csv"
asset_dir="assets"
train_file="$asset_dir/mnist_train.csv"
test_file="$asset_dir/mnist_test.csv"

if [ -f "$train_file" ] && [ -f "$test_file" ]; then
    echo "MNIST CSV files already exist in $asset_dir/."
    exit 0
fi

for command_name in curl unzip; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "error: $command_name is required" >&2
        exit 1
    fi
done

temporary_dir=$(mktemp -d "${TMPDIR:-/tmp}/nn-mnist.XXXXXX")
trap 'rm -rf "$temporary_dir"' EXIT HUP INT TERM

archive="$temporary_dir/mnist-in-csv.zip"
extracted="$temporary_dir/extracted"
mkdir -p "$extracted"

echo "Downloading MNIST in CSV format from Kaggle..."
curl --fail --location --retry 3 --output "$archive" "$dataset_url"
unzip -q -j "$archive" 'mnist_train.csv' 'mnist_test.csv' -d "$extracted"

remove_header() {
    csv_file=$1
    first_value=$(awk -F, 'NR == 1 { print $1; exit }' "$csv_file")

    if [ "$first_value" = "label" ]; then
        without_header="$csv_file.without-header"
        sed '1d' "$csv_file" > "$without_header"
        mv "$without_header" "$csv_file"
    fi
}

remove_header "$extracted/mnist_train.csv"
remove_header "$extracted/mnist_test.csv"

train_rows=$(wc -l < "$extracted/mnist_train.csv")
test_rows=$(wc -l < "$extracted/mnist_test.csv")
train_columns=$(awk -F, 'NR == 1 { print NF; exit }' "$extracted/mnist_train.csv")
test_columns=$(awk -F, 'NR == 1 { print NF; exit }' "$extracted/mnist_test.csv")

if [ "$train_rows" -ne 60000 ] || [ "$test_rows" -ne 10000 ] || \
   [ "$train_columns" -ne 785 ] || [ "$test_columns" -ne 785 ]; then
    echo "error: downloaded files do not have the expected MNIST CSV shape" >&2
    exit 1
fi

mkdir -p "$asset_dir"
mv "$extracted/mnist_train.csv" "$train_file"
mv "$extracted/mnist_test.csv" "$test_file"

echo "Downloaded $train_file (60,000 rows) and $test_file (10,000 rows)."
