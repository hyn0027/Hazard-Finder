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
    Construct a textual description of the software system from the configuration.
    """
    system_aim = config["system"]["system_aim"]
    use_cases = config["system"]["use_cases"]

    use_cases = [f"- {use_case}" for use_case in use_cases]
    use_cases = "\n".join(use_cases)

    return (
        f"The software system is designed to {system_aim}. "
        f"The system may be used in the following use cases:\n{use_cases} "
    )


def agent_function_description(config) -> str:
    """
    Construct a textual description of the agent's function from the configuration.
    """

    agent_function_aim = config["agent_function"]["function_aim"]
    tools = config["agent_function"]["tools"]
    use_cases = config["agent_function"]["use_cases"]

    tool_descriptions = ""

    for tool in tools:
        tool_name = tool["name"]
        tool_description = tool["description"]
        parameters = tool.get("parameters", [])
        response = tool.get("response", [])

        # Format parameters and responses as bullet-point lists
        if parameters:
            parameters = [
                f"- {param['name']}: {param['description']}" for param in parameters
            ]
            parameters = "\n".join(parameters)
        else:
            parameters = "None"

        if response:
            response = [f"- {resp['name']}: {resp['description']}" for resp in response]
            response = "\n".join(response)
        else:
            response = "None"

        tool_descriptions += (
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Parameters:\n{parameters}\n"
            f"Response:\n{response}\n"
            "\n"
        )

    # Format use cases as a bullet-point list
    use_cases = [f"- {use_case.strip()}" for use_case in use_cases]
    use_cases = "\n".join(use_cases)

    # Return a descriptive summary of the system
    return (
        f"The agent function is to {agent_function_aim}. "
        f"The agent may be used in the following use cases:\n{use_cases}\n\n"
        f"The agent can use the following tools:\n{tool_descriptions} "
    )
