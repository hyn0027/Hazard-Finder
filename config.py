import yaml
import logging

# Set up basic logging configuration to output messages to the console
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

# Define the order of processing steps in the ML system pipeline
STEP_ORDER = [
    "identify_stakeholders",
    "identify_values",
    "identify_losses",
    "identify_hazards",
    "consolidate_hazards",
    "divide_and_consolidate1",
    "divide_and_consolidate2",
    "divide_and_consolidate3",
]


def load_config(file_path="config.yml"):
    """
    Load configuration from a YAML file and determine which steps to skip
    based on the provided checkpoint.
    """

    with open(file_path, "r") as file:
        config = yaml.safe_load(file)

    # Check if a checkpoint is defined and valid
    if config["checkpoint"] is not None and config["checkpoint"] != "None":
        if config["checkpoint"] not in STEP_ORDER:
            raise ValueError(
                f"Invalid checkpoint value: {config['checkpoint']}. Must be one of {STEP_ORDER}"
            )

        # Determine which steps to skip based on the checkpoint
        skip_steps = []
        for step in STEP_ORDER:
            if step == config["checkpoint"]:
                break
            skip_steps.append(step)

        config["skip_steps"] = skip_steps
        logging.info(f"Skipping steps: {skip_steps}")
    else:
        config["skip_steps"] = []

    return config


def system_description(config) -> str:
    """
    Construct a textual description of the ML system from the configuration.
    """

    system_aim = config["ML_system"]["system_aim"]
    use_cases = config["ML_system"]["use_cases"]
    ML_purpose = config["ML_system"]["ML_purpose"]

    # Format use cases as a bullet-point list
    use_cases = [f"- {use_case}" for use_case in use_cases]
    use_cases = "\n".join(use_cases)

    # Return a descriptive summary of the system
    return (
        f"The software system is used to:\n{system_aim}\n\n"
        f"Use cases of this software system include:\n{use_cases}\n\n"
        f"There are ML models as components in this software system, they are used to to: \n{ML_purpose}"
    )
