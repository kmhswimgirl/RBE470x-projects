import sys
import os
sys.path.insert(0, '../../Bomberman')
sys.path.insert(1, '..')

# Test runner for all variants
# run specified variant a specified number of times and log the results

VARIANT = 3  # Change this to 1, 2, 3, 4, or 5 to select the variant to run
NUM_RUNS = 10  # Number of times to run the selected variant

CHARACTER_NAME = "me"  # Name of the character to track in the logs

# run the selected variant multiple times and log results
# look in the terminal output for results
# if the character wins then the message "me found the exit" will appear
# if the character dies then the message "me was killed"

import subprocess

# run the file corresponding to the selected variant

wins = 0
losses = 0

output_lines = []

for i in range(NUM_RUNS):
    process = subprocess.Popen(
        ["python3", f"variant{VARIANT}.py"],
        cwd="project1",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end="")           # print live
        output_lines.append(line)     # also save

    process.wait()

    # analyze the output
    for line in output_lines:
        if f"{CHARACTER_NAME} found the exit" in line:
            wins += 1
        elif f"{CHARACTER_NAME} was killed" in line:
            losses += 1
    
    # clear output lines for next run
    output_lines.clear()

print(f"\nOut of {NUM_RUNS} runs of variant {VARIANT}, {CHARACTER_NAME} won {wins} times and lost {losses} times.")