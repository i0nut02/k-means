import os
import subprocess

current_dir = os.path.abspath(".")

INPUT_FILE_FOLDER = os.path.join(current_dir, "test_files")
FOLDER_TO_RUN = {
    "sequential": os.path.join(current_dir, "sequential/bin/kmeans"),
    "cuda": os.path.join(current_dir, "cuda/bin/kmeans"),
    "cudaV2": os.path.join(current_dir, "cudaV2/bin/kmeans")
}

NODE_COUNTS = [1, 2, 4, 8]
THREAD_COUNTS = [1, 2, 4, 8, 16]

MAKE_FORLDERS = ["sequential", "cuda", "cudaV2", "mpi_openmp"]

FOLDER_TO_RUN.update({f"mpi_openmp_{nodes}_{threads}" : os.path.join(current_dir, "mpi_openmp/bin/kmeans") for nodes in NODE_COUNTS for threads in THREAD_COUNTS})

LOG_DIR = os.path.join(current_dir, "logs")
RESULTS_DIR = os.path.join(current_dir, "results")

NUM_CLUSTERS = [20]
CHANGES = 0
THRESHOLD = 0
ITERATIONS = 5000

RUNNING_SAMPLES = 10

LIMIT_ACTIVE_JOBS = 100

def get_active_jobs():
    """Returns the number of active jobs in the queue."""
    try:
        result = subprocess.run(["condor_q"], capture_output=True, text=True, check=True)
        return result.stdout.count("ID:")  # Counting job entries in condor_q output
    except subprocess.CalledProcessError:
        print("Error retrieving active jobs count.")
        return 0

def build_executables():
    """Runs 'make' in each folder to ensure the executables are compiled."""
    for model in MAKE_FORLDERS:
        print(f"Building {model}")
        subprocess.run(["make"], cwd=model, check=True)

def generate_and_submit_jobs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for model, executable in FOLDER_TO_RUN.items():
        num_threads = num_nodes = 0
        if model.startswith("mpi_openmp"):
            num_nodes, num_threads = tuple(map(int, model.split("_")[-2:]))

        results_folder = os.path.join(RESULTS_DIR, os.path.basename(model))
        os.makedirs(results_folder, exist_ok=True)

        for input_file in os.listdir(INPUT_FILE_FOLDER):
            input_path = os.path.join(INPUT_FILE_FOLDER, input_file)
            
            for cluster in NUM_CLUSTERS:
                input_result_folder = os.path.join(results_folder, input_file + f"_clusters_{cluster}")
                os.makedirs(input_result_folder, exist_ok=True)

                for it in range(RUNNING_SAMPLES):
                    while get_active_jobs() >= LIMIT_ACTIVE_JOBS:
                        print("Job limit reached. Waiting before submitting more jobs...")
                        time.sleep(30)  # Wait before checking again

                    # Define result output files
                    output_file = os.path.join(input_result_folder, f"{it+1}.out")
                    log_file = os.path.join(input_result_folder, f"{it+1}.log")

                    # Condor submit file path
                    condor_file = f"{model}_{input_file}_clusters_{cluster}_iteration_{it}.sub"
                    
                    # Create condor submission script based on model type
                    with open(condor_file, "w") as f:
                        if model == "sequential":
                            f.write(f"""universe = vanilla
                                    log = {LOG_DIR}/{model}_job.log
                                    output = {LOG_DIR}/{model}_job.out
                                    error = {LOG_DIR}/{model}_job.err
                                    executable = {executable}
                                    arguments = {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {output_file} {log_file}
                                    getenv = True
                                    queue
                                    """)
                        elif model.startswith("cuda"):
                            f.write(f"""universe = vanilla
                                    log = logs/{model}_job.log
                                    output = logs/{model}_job.out
                                    error = logs/{model}_job.err
                                    request_gpus = 1
                                    executable = {executable}
                                    arguments = {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {output_file} {log_file}
                                    getenv = True
                                    queue
                                    """)
                        elif model.startswith("mpi_openmp"):
                            f.write(f"""universe = parallel
                                    executable = ./mpi_openmp/openmpiscript.sh
                                    arguments = {executable} {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {output_file} {log_file} {num_threads}
                                    should_transfer_files = YES
                                    transfer_input_files = {executable}
                                    when_to_transfer_output = on_exit_or_evict
                                    output = logs/out.$(NODE)
                                    error = logs/err.$(NODE)
                                    log = logs/log
                                    machine_count = {num_nodes}
                                    request_cpus = {num_threads}
                                    getenv = True
                                    queue
                                    """)
                    # Submit the job
                    print(f"Submitting {condor_file} for {model} (clusters={cluster}, input={input_file})")
                    subprocess.run(["condor_submit", condor_file], check=True)

if __name__ == "__main__":
    build_executables()
    generate_and_submit_jobs()
