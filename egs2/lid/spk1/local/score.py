import argparse

# score.py

def read_file(file_path):
    """
    Reads a file and returns a dictionary with key and lid.
    Each line in the file should be in the format: key lid
    """
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                key, lid = parts
                data[key] = lid
            else:
                raise ValueError(f"Invalid line format: {line}")
    return data

def score(pred_file, target_file, results_file):
    """
    Calculates the accuracy by comparing the predicted and target lids.
    """
    pred_data = read_file(pred_file)
    target_data = read_file(target_file)

    if set(pred_data.keys()) != set(target_data.keys()):
        raise ValueError("Keys in pred and target files do not match.")

    total = len(pred_data)
    correct = 0
    incorrect_predictions = {}

    for key in pred_data:
        if pred_data[key] == target_data[key]:
            correct += 1
        else:
            incorrect_predictions[key] = (pred_data[key], target_data[key])

    accuracy = correct / total

    if incorrect_predictions:
        with open(results_file, 'w') as f:
            f.write(f"Accuracy: {accuracy:.2%}\n")
            for key, (pred, target) in incorrect_predictions.items():
                f.write(f"Key: {key}, Target: {target}, Predicted: {pred}\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate accuracy of predictions.")
    parser.add_argument("--exp_dir", required=True, help="Path to the experiment directory.")
    parser.add_argument("--test_set_name", required=True, help="Name of the test set.")
    args = parser.parse_args()

    pred = f"{args.exp_dir}/inference/{args.test_set_name}/{args.test_set_name}_lids"
    target = f"{args.exp_dir}/../../dump/raw/{args.test_set_name}/utt2spk"
    results = f"{args.exp_dir}/inference/{args.test_set_name}/results"

    score(pred, target, results)