import sys
import json


# Utility function to pause program execution based on user input
def pause_execution():
    # Prompts the user to continue or exit the program
    while True:
        response = input("Do you want to continue? (Y/N): ")
        if response.lower() == "y":
            break  # Continue execution
        elif response.lower() == "n":
            sys.exit(0)  # Exit the program
        else:
            print("Invalid input. Please enter Y or N.")  # Re-prompt on invalid input
            continue


# Utility function to save a Python object to a JSON file
def save_to_json(content, path):
    with open(path, "w") as file:
        json.dump(
            content, file, indent=4
        )  # Write JSON with indentation for readability


# Utility function to load content from a JSON file into a Python object
def load_from_json(path):
    with open(path, "r") as file:
        content = json.load(file)  # Read and parse the JSON file
    return content
