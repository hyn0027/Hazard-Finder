from OpenAIChatHelper import ChatCompletionEndPoint
from OpenAIChatHelper.message import (
    SubstitutionDict,
    MessageList,
    DevSysUserMessage,
    TextContent,
)
from embedding import get_embedding
import numpy as np
from sklearn.cluster import KMeans
import logging
import random

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


# Identify stakeholders using chatbot, based on the system description
def identify_stakeholders(
    chatbot: ChatCompletionEndPoint, substitution_dict: SubstitutionDict
):
    message_list = MessageList()

    # Instruction message for the chatbot
    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the description of the software system, identify and list all potential stakeholders, both direct and indirect. "
                "Consider any individuals, groups, organizations, or entities that may create, interact with, be affected by, or influence the system in any way. Think as broadly as you can. "
                "Ensure your response is as comprehensive as possible by considering stakeholders across various levels. "
                "Format your response as follows:\n"
                "1. Stakeholder Name - Description of their role and responsibilities.\n"
                "2. Stakeholder Name - Description of their role and responsibilities.\n"
                "3. Stakeholder Name - Description of their role and responsibilities.\n"
                "4. Stakeholder Name - Description of their role and responsibilities.\n"
                "5. Stakeholder Name - Description of their role and responsibilities.\n"
                "6. Stakeholder Name - Description of their role and responsibilities.\n"
                "7. Stakeholder Name - Description of their role and responsibilities.\n"
                "8. Stakeholder Name - Description of their role and responsibilities.\n"
                "9. Stakeholder Name - Description of their role and responsibilities.\n"
                "...\n"
                "...\n"
            ),
        )
    )

    # Placeholder for system description, to be replaced via substitution_dict
    message_list.add_message(
        DevSysUserMessage(
            "user",
            TextContent(
                'Software System Description: \n"""{system_description}\n"""\nStakeholders:\n'
            ),
        )
    )

    # Get chatbot response
    res, meta = chatbot.completions(
        message_list, substitution_dict=substitution_dict, temperature=0.0
    )

    # Parse returned list of stakeholders into name and description
    stakeholder_content: TextContent = res[0][0]
    stakeholders = stakeholder_content.split_ordered_list()
    stakeholder_list = []
    for stakeholder in stakeholders:
        stakeholder = stakeholder.strip()
        stake_holder_name, stake_holder_description = stakeholder.split(" - ", 1)
        stakeholder_list.append(
            {"name": stake_holder_name, "description": stake_holder_description}
        )
    stakeholders = stakeholder_list
    return stakeholders


# Identify values for each stakeholder using the chatbot
def identify_values(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    stakeholders: list,
):
    message_list = MessageList()

    # Instruction message to the chatbot
    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the system description, "
                "identify the high-level values and goals that the given stakeholder may expect from the system. "
                "Format your response as follows:\n"
                "1. A Short phrase describing value or goal 1\n"
                "2. A Short phrase describing value or goal 2\n"
                "3. A Short phrase describing value or goal 3\n"
                "4. A Short phrase describing value or goal 4\n"
                "... \n"
                "... \n"
            ),
        )
    )

    # User input template with placeholders
    message_list.add_message(
        DevSysUserMessage(
            "user",
            TextContent(
                "System Description:\n"
                "{system_description}\n\n"
                "Stakeholder:\n"
                "{stakeholder}\n\n"
                "Values and Goals:\n"
            ),
        )
    )

    # Iterate through each stakeholder to get their associated values/goals
    for i in range(len(stakeholders)):
        item = stakeholders[i]
        substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
        res, meta = chatbot.completions(
            message_list, substitution_dict=substitution_dict, temperature=0.0
        )
        values_content: TextContent = res[0][0]
        value = values_content.split_ordered_list()
        value = [val.strip() for val in value]
        logging.info(f"Values and Goals for {item['name']}:")
        for val in value:
            logging.info(f"\t- {val}")
        logging.info(f"{'*' * 5}")
        stakeholders[i]["values"] = value

    return stakeholders


# Identify potential losses from values using chatbot
def identify_losses(
    chatbot: ChatCompletionEndPoint, substitution_dict: SubstitutionDict, values: list
):
    message_list = MessageList()

    # Instruction for converting a value into a loss
    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Using the system description and a given high-level value or goal of the stakeholder, "
                "reverse the value or goal into a corresponding high-level loss. "
                "Format your response as follows:\n"
                "A Short phrase describing the loss"
            ),
        )
    )

    # User message template
    message_list.add_message(
        DevSysUserMessage(
            "user",
            TextContent(
                "System Description:\n"
                "{system_description}\n\n"
                "Stakeholder:\n"
                "{stakeholder}\n\n"
                "Values pr goal:\n"
                "{value}\n\n"
                "Loss:\n"
            ),
        )
    )

    # Loop through values and convert each into a potential loss
    for i in range(len(values)):
        item = values[i]
        substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
        logging.info(f"Identifying losses for {item['name']}")
        for val in item["values"]:
            substitution_dict["value"] = val
            res, meta = chatbot.completions(
                message_list, substitution_dict=substitution_dict, temperature=0.0
            )
            loss_content: TextContent = res[0][0]
            loss = loss_content.text.strip()
            logging.info(f"\tLoss for {val} is: {loss}")
            if "losses" not in item:
                item["losses"] = []
            item["losses"].append(loss)
        logging.info(f"{'*' * 5}")
        values[i] = item
    return values


# Identify hazards that could lead to each loss
def identify_hazards(
    chatbot: ChatCompletionEndPoint, substitution_dict: SubstitutionDict, losses: list
):
    message_list = MessageList()

    # Instruction for identifying hazards
    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Given a description of a system, and a specific loss of a stakeholder, "
                "identify and list potential states and conditions that could directly lead to this loss under worst-case conditions. "
                "Provide concise, standalone descriptions of these states and conditions. "
                "Do not include any cause, explanation, result, or solution to the state or condition. "
                "Format your response as follows:\n"
                "1. State or condition 1\n"
                "2. State or condition 2\n"
                "3. State or condition 3\n"
                "... \n"
                "... \n"
            ),
        )
    )

    # Template message for the chatbot
    message_list.add_message(
        DevSysUserMessage(
            "user",
            TextContent(
                "System Description:\n"
                "{system_description}\n\n"
                "Stakeholder:\n"
                "{stakeholder}\n\n"
                "Loss:\n"
                "{loss}\n\n"
                "States and Conditions that could Lead to this Loss (Worst-Case Scenarios):\n"
            ),
        )
    )

    # Loop over each loss and collect hazards
    for i in range(len(losses)):
        item = losses[i]
        substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
        logging.info(f"Identifying hazards for {item['name']}")
        item["hazards"] = {}
        for j in range(len(item["losses"])):
            loss = item["losses"][j]
            substitution_dict["loss"] = loss
            res, meta = chatbot.completions(
                message_list, substitution_dict=substitution_dict, temperature=0.0
            )
            hazard_content: TextContent = res[0][0]
            hazard = hazard_content.split_ordered_list()
            hazard = [h.strip() for h in hazard]
            logging.info(f"Hazards for {loss}:")
            for h in hazard:
                logging.info(f"\t- {h}")
            logging.info(f"{'*' * 5}")
            item["hazards"][loss] = hazard
        losses[i] = item
    return losses


# Consolidate all hazard statements by clustering and summarizing them
def consolidate_hazards(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    hazards_comprehensive: list,
):
    hazard_num = 0
    for item in hazards_comprehensive:
        for loss, hazards in item["hazards"].items():
            hazard_num += len(hazards)
    logging.info(f"Total number of hazards: {hazard_num}")

    hazard_list = []
    for item in hazards_comprehensive:
        hazard_list_per_item = []
        for loss, hazards in item["hazards"].items():
            hazard_list_per_item.extend(hazards)
        hazard_list_per_item = consolidate_hazard_list(
            chatbot, substitution_dict, hazard_list_per_item, n_clusters=20
        )
        hazard_list.extend(hazard_list_per_item)
    logging.info(f"Total number of consolidated hazards: {len(hazard_list)}")
    return hazard_list


# Break down hazard list into segments, consolidate each segment
def divide_and_consolidate(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    hazard_list: list,
    n_clusters=20,
    segment_size=200,
):
    random.shuffle(hazard_list)  # Randomize to avoid bias in ordering
    res = []
    for i in range(0, len(hazard_list), segment_size):
        segment = hazard_list[i : i + segment_size]
        segment = consolidate_hazard_list(
            chatbot, substitution_dict, segment, n_clusters=n_clusters
        )
        res.extend(segment)
    return res


# Core function to group similar hazards using embeddings and clustering
def consolidate_hazard_list(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    hazard_list: list,
    n_clusters=20,
):
    if n_clusters != 1:
        logging.info("Getting embeddings for hazards...")
        embeddings = [get_embedding(hazard) for hazard in hazard_list]

        logging.info("Performing clustering on embeddings...")
        embeddings = np.array(embeddings)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
        labels = kmeans.labels_
        hazard_clusters = {}
        for i, label in enumerate(labels):
            label = str(label)
            if label not in hazard_clusters:
                hazard_clusters[label] = []
            hazard_clusters[label].append(hazard_list[i])
    else:
        hazard_clusters = {"0": hazard_list}

    # Log clusters before consolidation
    for cluster, hazards in hazard_clusters.items():
        logging.info(f"Cluster {cluster}:")
        for hazard in hazards:
            logging.info(f"\t- {hazard}")
        logging.info(f"{'*' * 5}")

    # Prepare messages for merging similar hazard statements
    message_list = MessageList()
    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the system description, review the list of state or conditions and identify those that have close meanings. "
                "Merge similar state or conditions into a single concise entry."
                "Each entry should be a concise, standalone description of a state or condition. "
                "Do not include any cause, explanation, result, or solution to the state or condition. "
                "Format your response as follows:\n"
                "1. Merged State or Condition 1\n"
                "2. Merged State or Condition 2\n"
                "... \n"
            ),
        )
    )
    message_list.add_message(
        DevSysUserMessage(
            "user",
            TextContent(
                "System Description:\n"
                "{system_description}\n\n"
                "State or Condition List:\n"
                "{hazard_list}\n\n"
                "Merged States or Conditions:\n"
            ),
        )
    )

    res_list = []
    for cluster, hazards in hazard_clusters.items():
        hazards = [f"- {hazard}" for hazard in hazards]
        substitution_dict["hazard_list"] = "\n".join(hazards)
        res, meta = chatbot.completions(
            message_list, substitution_dict=substitution_dict, temperature=0.0
        )
        consolidated_hazards: TextContent = res[0][0]
        consolidated_hazards = consolidated_hazards.split_ordered_list()
        consolidated_hazards = [h.strip() for h in consolidated_hazards]
        logging.info(f"Consolidated Hazards for Cluster {cluster}:")
        for h in consolidated_hazards:
            logging.info(f"\t- {h}")
        logging.info(f"{'*' * 5}")
        res_list.extend(consolidated_hazards)
    return res_list
