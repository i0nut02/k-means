import os

# Define paths
RESULTS = "./results"
CORRECT = "sequential"

# Helper function to get file content
def get_file_content(file_path):
    with open(file_path, "r") as file:
        return file.read()

# Assert equality of file contents
def assert_equal(content1, content2, file1, file2):
    if content1 != content2:
        raise AssertionError(f"Contents of {file1} and {file2} do not match.")

def main():
    # Ensure the correct folder exists
    correct_folder_path = os.path.join(RESULTS, CORRECT)
    if not os.path.exists(correct_folder_path):
        raise FileNotFoundError(f"Correct folder '{CORRECT}' not found in RESULTS.")

    # Iterate over each folder in RESULTS
    for folder in os.listdir(RESULTS):
        folder_path = os.path.join(RESULTS, folder)

        # Skip the CORRECT folder or non-directory files
        if folder == CORRECT or not os.path.isdir(folder_path):
            continue

        # Iterate over files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Check for .out files
            if file_name.endswith(".out"):
                correct_file_path = os.path.join(correct_folder_path, file_name)

                # Ensure the corresponding file exists in the correct folder
                if not os.path.exists(correct_file_path):
                    raise FileNotFoundError(f"Correct file '{file_name}' not found in '{CORRECT}' folder.")

                # Compare file contents
                folder_content = get_file_content(file_path)
                correct_content = get_file_content(correct_file_path)
                assert_equal(folder_content, correct_content, file_path, correct_file_path)

    print("All .out files in non-correct folders match the correct folder!")

if __name__ == "__main__":
    main()
