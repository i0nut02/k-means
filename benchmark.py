import os
import subprocess

INPUT_FILE_FOLDER = "./test_files"
FOLDER_TO_RUN = ["./sequential"] #["./sequential", "./cuda", "./mpi_omp"]
NUM_CLUSTERS = [3, 10, 20]
CHANGES = 0
THRESHOLD = 0
ITERATIONS = 300

def run_benchmark():
    # Create results directory if it doesn't exist
    print("Current working directory:", os.getcwd())
    print("Contents:", os.listdir())
    os.makedirs("results", exist_ok=True)
    
    # Loop through each folder to run
    for run_model in FOLDER_TO_RUN:
        # Run make command in each folder

        subprocess.run(["make"], cwd=run_model, check=True)
        
        # Create a subdirectory in results for each folder
        folder_results = os.path.join("results", os.path.basename(run_model))
        os.makedirs(folder_results, exist_ok=True)
        
        # Loop through each input file
        for input_file in os.listdir(INPUT_FILE_FOLDER):
            input_path = os.path.join(INPUT_FILE_FOLDER, input_file)
            
            # Loop through each number of clusters
            for cluster in NUM_CLUSTERS:
                # Construct the output path
                output_file = f"res_{input_file}_clusters_{cluster}.txt"
                output_path = os.path.join(folder_results, output_file) # ./results/{run_model/{output_file}}
                
                # Run the k-means executable with the required arguments
                executable = os.path.join(run_model, "bin", "kmeans")
                subprocess.run([executable, input_path, str(cluster), str(ITERATIONS), 
                                str(CHANGES), str(THRESHOLD), output_path], check=True)

if __name__ == "__main__":
    run_benchmark()
