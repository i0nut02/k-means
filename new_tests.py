import os

TEST_FILES_FOLDER = os.path.join(os.path.abspath("."), "test_files")
TEST_TO_SCALE = "input100D2.inp"

def add_tests():
    i = 1
    
    # Read the original content
    input_path = os.path.join(TEST_FILES_FOLDER, TEST_TO_SCALE)
    with open(input_path, 'r') as f:
        content = f.read()
    
    for _ in range(3):
        # Calculate new number of points
        i = i << 1
        
        # Create new filename
        new_filename = f"input100D2X{i}.inp"
        output_path = os.path.join(TEST_FILES_FOLDER, new_filename)
        
        # Write doubled content to new file
        with open(output_path, 'w') as f:
            f.write(content)
            f.write(content)
        
        # Update content for next iteration
        content = content + content

if __name__ == "__main__":
    add_tests()