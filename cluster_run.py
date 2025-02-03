import os
import subprocess

INPUT_FILE_FOLDER = "./test_files"
FOLDER_TO_RUN = {
    "sequential": "./sequential/bin/kmeans",
    "cuda": "./cuda/bin/kmeans",
    "mpi_openmp": "./mpi_openmp/bin/kmeans"
}
NUM_CLUSTERS = [3]
CHANGES = 0
THRESHOLD = 0
ITERATIONS = 300

def build_executables():
    """Runs 'make' in each folder to ensure the executables are compiled."""
    for model, _ in FOLDER_TO_RUN.items():
        print(f"Building {model}")
        subprocess.run(["make"], cwd=model, check=True)

def generate_and_submit_jobs():
    os.makedirs("logs", exist_ok=True)

    for model, executable in FOLDER_TO_RUN.items():
        for input_file in os.listdir(INPUT_FILE_FOLDER):
            input_path = os.path.join(INPUT_FILE_FOLDER, input_file)
            
            for cluster in NUM_CLUSTERS:
                # Define result output files
                output_file = f"results/{model}/{input_file}_clusters_{cluster}.out"
                log_file = f"results/{model}/{input_file}_clusters_{cluster}.log"
                
                # Condor submit file path
                condor_file = f"{model}_{input_file}_clusters_{cluster}.sub"
                
                # Create condor submission script based on model type
                with open(condor_file, "w") as f:
                    if model == "sequential":
                        f.write(f"""universe = vanilla
                                log = logs/{model}_job.log
                                output = logs/{model}_job.out
                                error = logs/{model}_job.err
                                executable = {executable}
                                arguments = {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {output_file} {log_file}
                                getenv = True
                                queue
                                """)
                    elif model == "cuda":
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
                    elif model == "mpi_openmp":
                        f.write(f"""universe = parallel
                                executable = mpi_openmp/run_kmeans.sh
                                arguments = {input_path} {cluster} {ITERATIONS} {CHANGES} {THRESHOLD} {output_file} {log_file}
                                should_transfer_files = YES
                                transfer_input_files = {executable}
                                when_to_transfer_output = on_exit_or_evict
                                output = logs/out.$(NODE)
                                error = logs/err.$(NODE)
                                log = logs/log
                                machine_count = 4
                                request_cpus = 8
                                getenv = True
                                queue
                                """)

                # Submit the job
                print(f"Submitting {condor_file} for {model} (clusters={cluster}, input={input_file})")
                subprocess.run(["condor_submit", condor_file], check=True)
            break # to do only one file

if __name__ == "__main__":
    build_executables()
    generate_and_submit_jobs()
