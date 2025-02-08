import os

# Define paths
RESULTS = "./results"
CORRECT = "sequential"

def get_file_content(file_path):
    with open(file_path, "r") as file:
        return file.read()

def assert_equal(content1, content2, file1, file2):
    if content1 != content2:
        raise AssertionError(f"Contents of {file1} and {file2} do not match.")

def main():
    # Ensure the correct folder exists
    correct_path = os.path.join(RESULTS, CORRECT)
    if not os.path.exists(correct_path):
        raise FileNotFoundError(f"Correct folder '{CORRECT}' not found in RESULTS.")

    # Iterate over model types in RESULTS
    for model_type in os.listdir(RESULTS):
        model_path = os.path.join(RESULTS, model_type)
        
        # Skip if not a directory or if it's the correct folder
        if not os.path.isdir(model_path) or model_type == CORRECT:
            continue

        # Iterate over test names in model type
        for test_name in os.listdir(model_path):
            test_path = os.path.join(model_path, test_name)
            
            if not os.path.isdir(test_path):
                continue

            # Find corresponding test folder in correct path
            correct_test_path = os.path.join(correct_path, test_name)
            if not os.path.exists(correct_test_path):
                raise FileNotFoundError(f"Test folder '{test_name}' not found in '{CORRECT}' folder.")

            # Iterate over .out files in test folder
            for file_name in os.listdir(test_path):
                if file_name.endswith(".out"):
                    file_path = os.path.join(test_path, file_name)
                    correct_file_path = os.path.join(correct_test_path, file_name)

                    # Ensure the corresponding file exists in correct folder
                    if not os.path.exists(correct_file_path):
                        raise FileNotFoundError(
                            f"File '{file_name}' not found in correct test folder: {correct_test_path}"
                        )

                    # Compare file contents
                    test_content = get_file_content(file_path)
                    correct_content = get_file_content(correct_file_path)
                    assert_equal(test_content, correct_content, file_path, correct_file_path)

    print("\nAll .out files match their corresponding files in the sequential folder!")

if __name__ == "__main__":
    main()