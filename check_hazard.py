from OpenAIChatHelper import ChatCompletionEndPoint
from OpenAIChatHelper.message import (
    SubstitutionDict,
    MessageList,
    DevSysUserMessage,
    TextContent,
)
from config import load_config, system_description, agent_function_description
from utils import pause_execution, save_to_json, load_from_json
import logging
import asyncio
import random
from copy import deepcopy


# Configure logging format and level
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


sample_size = 100


async def main():

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

    # load hazards
    full_list = load_from_json("hazards.json")

    hazard_list = []

    for item in full_list:
        stakeholder = item["name"]
        stakeholder_description = item["description"]
        hazards_dict = item["hazards"] if "hazards" in item else {}

        for loss in hazards_dict:
            hazards = hazards_dict[loss]
            for hazard in hazards:
                hazard_list.append(
                    {
                        "stakeholder": stakeholder,
                        "stakeholder_description": stakeholder_description,
                        "loss": loss,
                        "hazard": hazard,
                    }
                )

    # randomly sample hazards for testing
    selected_hazards = random.sample(hazard_list, sample_size)

    message_list = MessageList()

    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the description of a software system and a specific function within that system that is enabled by LLM-powered agents, "
                "given a specific stakeholder, and given a specific loss that the stakeholder does not expect from the function, "
                "here is a specific hazard that could lead to that loss. "
                "Now consider ways to prevent that hazard. "
                "If it can be solved by adding checks in the tool API, respond with 'Yes', and a brief explanation. "
                "If it cannot be solved by adding checks in the tool API, respond with 'No', and a brief explanation, plus potential alternative mitigations. \n\n"
                "Format your responses as follows:\n"
                "Yes/No\n"
                "Explanation: [Explanation]\n"
            ),
        )
    )

    # Template message for the chatbot
    message_list.add_message(
        DevSysUserMessage(
            "user",
            TextContent(
                "Software System Description: \n"
                "{system_description}\n\n"
                "Agent Function Description: \n"
                "{agent_function}\n\n"
                "Stakeholder:\n"
                "{stakeholder}\n\n"
                "Loss:\n"
                "{loss}\n\n"
                "Hazard:\n"
                "{hazard}\n\n"
            ),
        )
    )

    total_num = len(selected_hazards)
    semaphore = asyncio.Semaphore(100)

    async def process_hazard(id, substitution_dict, message_list, chatbot, hazard_item):
        async with semaphore:
            substitution_dict["stakeholder"] = hazard_item["stakeholder"]
            substitution_dict["stakeholder_description"] = hazard_item[
                "stakeholder_description"
            ]
            substitution_dict["loss"] = hazard_item["loss"]
            substitution_dict["hazard"] = hazard_item["hazard"]
            logging_text = (
                f"Processing hazard {id+1}/{total_num}:\n"
                f"Stakeholder: {hazard_item['stakeholder']}\n"
                f"Loss: {hazard_item['loss']}\n"
                f"Hazard: {hazard_item['hazard']}\n"
            )

            res, meta = await chatbot.completions(
                message_list,
                substitution_dict=substitution_dict,
            )

            res_content: TextContent = res[0][0]
            res_text = res_content.text.strip()
            logging_text += f"Response:\n{res_text}\n"
            logging.info(logging_text)
            is_yes = res_text.lower().startswith("yes")
            hazard_item["api_check_possible"] = is_yes
            hazard_item["explanation"] = res_text
            return hazard_item

    tasks = []

    for index, hazard_item in enumerate(selected_hazards):
        tasks.append(
            asyncio.create_task(
                process_hazard(
                    index,
                    deepcopy(substitution_dict),
                    deepcopy(message_list),
                    chatbot,
                    hazard_item,
                )
            )
        )

    results = await asyncio.gather(*tasks)
    save_to_json(results, "hazard_api_check_results.json")


# Entry point for the script
if __name__ == "__main__":
    asyncio.run(main())
