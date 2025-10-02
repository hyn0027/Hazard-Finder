from OpenAIChatHelper import ChatCompletionEndPoint
from OpenAIChatHelper.message import (
    SubstitutionDict,
)
from config import load_config, system_description, agent_function_description
from steps import (
    identify_stakeholders,
    identify_values,
    identify_losses,
    identify_hazards,
    consolidate_hazards,
    divide_and_consolidate,
)
from utils import pause_execution, save_to_json, load_from_json
import logging
from filter import filter_value, filter_loss
import asyncio

# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


async def main():
    # Load configuration settings
    config = load_config()

    # Initialize substitution dictionary for prompt templating
    substitution_dict = SubstitutionDict()

    # Create a chatbot interface using the specified model
    chatbot = ChatCompletionEndPoint(default_model=config["chatbot"]["model"])

    # Generate and log system description, and store in the substitution dictionary
    system_description_message = system_description(config)
    agent_function = agent_function_description(config)
    logging.info(f"System description: {system_description_message}")
    logging.info(f"Agent function description: {agent_function}")
    substitution_dict["system_description"] = system_description_message
    substitution_dict["agent_function"] = agent_function
    pause_execution()

    # Step 1: Identify stakeholders unless skipped via config
    if "identify_stakeholders" not in config["skip_steps"]:
        stakeholders = identify_stakeholders(chatbot, substitution_dict)
        for stakeholder in stakeholders:
            logging.info(
                f"Stakeholder: {stakeholder['name']} - {stakeholder['description']}"
            )
        save_to_json(stakeholders, "stakeholders.json")
        logging.info("Stakeholders saved to stakeholders.json")
        pause_execution()
    else:
        stakeholders = load_from_json("stakeholders.json")
        logging.info("Stakeholders loaded from stakeholders.json")

    if "identify_values" not in config["skip_steps"]:
        values = await identify_values(
            chatbot, substitution_dict, stakeholders, dropout=config.get("dropout", 0.0)
        )
        save_to_json(values, "values.json")
        logging.info("Values saved to values.json")
        pause_execution()
    else:
        values = load_from_json("values.json")
        logging.info("Values loaded from values.json")

    if "filter_values" not in config["skip_steps"]:
        values = await filter_value(
            chatbot, substitution_dict, values, dropout=config.get("dropout", 0.0)
        )
        save_to_json(values, "filtered_values.json")
        logging.info("Filtered values saved to filtered_values.json")
        pause_execution()
    else:
        values = load_from_json("filtered_values.json")
        logging.info("Filtered values loaded from filtered_values.json")

    if "identify_losses" not in config["skip_steps"]:
        losses = await identify_losses(
            chatbot, substitution_dict, values, dropout=config.get("dropout", 0.0)
        )
        save_to_json(losses, "losses.json")
        logging.info("Losses saved to losses.json")
        pause_execution()
    else:
        losses = load_from_json("losses.json")
        logging.info("Losses loaded from losses.json")

    if "filter_losses" not in config["skip_steps"]:
        losses = await filter_loss(chatbot, substitution_dict, losses)
        save_to_json(losses, "filtered_losses.json")
        logging.info("Filtered losses saved to filtered_losses.json")
        pause_execution()
    else:
        losses = load_from_json("filtered_losses.json")
        logging.info("Filtered losses loaded from filtered_losses.json")

    if "identify_hazards" not in config["skip_steps"]:
        hazards = await identify_hazards(
            chatbot, substitution_dict, losses, dropout=config.get("dropout", 0.0)
        )
        save_to_json(hazards, "hazards.json")
        logging.info("Hazards saved to hazards.json")
        pause_execution()
    else:
        hazards = load_from_json("hazards.json")
        logging.info("Hazards loaded from hazards.json")

    return

    # Step 5: Consolidate hazards unless skipped
    if "consolidate_hazards" not in config["skip_steps"]:
        consolidated_hazards = consolidate_hazards(chatbot, substitution_dict, hazards)
        pause_execution()
        save_to_json(consolidated_hazards, "consolidated_hazards.json")
        logging.info("Consolidated hazards saved to consolidated_hazards.json")
    else:
        consolidated_hazards = load_from_json("consolidated_hazards.json")
        logging.info("Consolidated hazards loaded from consolidated_hazards.json")

    # Step 6: First round of divide and consolidate
    if "divide_and_consolidate1" not in config["skip_steps"]:
        consolidate_hazards1 = divide_and_consolidate(
            chatbot,
            substitution_dict,
            consolidated_hazards,
            n_clusters=10,
            segment_size=min(100, len(consolidated_hazards)),
        )
        pause_execution()
        save_to_json(consolidate_hazards1, "consolidate_hazards1.json")
        logging.info("Consolidated hazards saved to consolidate_hazards1.json")
    else:
        consolidate_hazards1 = load_from_json("consolidate_hazards1.json")
        logging.info("Consolidated hazards loaded from consolidate_hazards1.json")

    # Step 7: Second round of divide and consolidate
    if "divide_and_consolidate2" not in config["skip_steps"]:
        consolidate_hazards2 = divide_and_consolidate(
            chatbot,
            substitution_dict,
            consolidate_hazards1,
            n_clusters=5,
            segment_size=min(80, len(consolidate_hazards1)),
        )
        pause_execution()
        save_to_json(consolidate_hazards2, "consolidate_hazards2.json")
        logging.info("Consolidated hazards saved to consolidate_hazards2.json")
    else:
        consolidate_hazards2 = load_from_json("consolidate_hazards2.json")
        logging.info("Consolidated hazards loaded from consolidate_hazards2.json")

    # Step 8: Third round of divide and consolidate
    if "divide_and_consolidate3" not in config["skip_steps"]:
        consolidate_hazards3 = divide_and_consolidate(
            chatbot,
            substitution_dict,
            consolidate_hazards2,
            n_clusters=5,
            segment_size=min(80, len(consolidate_hazards2)),
        )
        pause_execution()
        save_to_json(consolidate_hazards3, "consolidate_hazards3.json")
        logging.info("Consolidated hazards saved to consolidate_hazards3.json")
    else:
        consolidate_hazards3 = load_from_json("consolidate_hazards3.json")
        logging.info("Consolidated hazards loaded from consolidate_hazards3.json")


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
