#!/bin/bash

# Check if the user provided a project number as input
if [ $# -eq 0 ]; then
    echo "Please provide the project number as an argument."
    exit 1
fi

# Get the project number from the command line argument
project_num="$1"

# Compile the C++ program
g++ -o project${project_num} project${project_num}.cpp -lm -fopenmp

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Running project${project_num}..."
    # Run the compiled executable
    ./project${project_num}
else
    echo "Compilation failed. Please check your code."
fi

