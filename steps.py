from concurrent.futures import ThreadPoolExecutor
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
import asyncio
from copy import deepcopy

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


# Identify stakeholders using chatbot, based on the system description
async def identify_stakeholders(
    chatbot: ChatCompletionEndPoint, substitution_dict: SubstitutionDict
):
    message_list = MessageList()

    # Instruction message for the chatbot
    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the description of a software system and a specific LLM-powered agents within that system, "
                "identify and list stakeholders related to the agent. "
                "The stakeholders should be related to the agent, rather than the overall system. \n\n"
                "Use umbrella terms to represent groups of stakeholders when appropriate. \n\n"
                "Format your response as follows:\n"
                "1. Stakeholder Name - Short description of the stakeholder and how they are related to the agent\n"
                "2. Stakeholder Name - Short description of the stakeholder and how they are related to the agent\n"
                "...\n"
            ),
        )
    )

    # Placeholder for system description, to be replaced via substitution_dict
    message_list.add_message(
        DevSysUserMessage(
            "user",
            TextContent(
                "Software System Description: \n"
                "{system_description}\n\n"
                "Agent Description: \n"
                "{agent_function}\n\n"
                "Stakeholders:\n"
            ),
        )
    )

    # Get chatbot response
    res, meta = await chatbot.completions(
        message_list,
        substitution_dict=substitution_dict,
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
async def identify_values(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    stakeholders: list,
    dropout: float = 0.0,
):
    message_list = MessageList()

    # Instruction message to the chatbot
    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the description of a software system and a specific LLM-powered agents within that system, "
                "and considering a particular stakeholder associated with that agent, "
                "identify the core high-level, abstract values that the stakeholder expects from the agent. "
                "Each value should focus on a single topic and avoid combining multiple aspects. "
                "The values should be related to the agent, rather than the overall system. "
                "\n\n"
                "Format your response as follows:\n"
                "1. A Short phrase describing value 1\n"
                "2. A Short phrase describing value 2\n"
                "3. A Short phrase describing value 3\n"
                "4. A Short phrase describing value 4\n"
                "5. A Short phrase describing value 5\n"
                "... \n"
            ),
        )
    )

    # User input template with placeholders
    message_list.add_message(
        DevSysUserMessage(
            "user",
            TextContent(
                "Software System Description: \n"
                "{system_description}\n\n"
                "Agent Description: \n"
                "{agent_function}\n\n"
                "Stakeholder:\n"
                "{stakeholder}\n\n"
                "Values and Goals:\n"
            ),
        )
    )

    semaphore = asyncio.Semaphore(100)

    async def fetch_values(
        i, item, message_list, substitution_dict, chatbot: ChatCompletionEndPoint
    ):
        if dropout > 0 and random.random() < dropout and i > 5:
            return None
        async with semaphore:
            substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
            res, meta = await chatbot.completions(
                message_list,
                substitution_dict=substitution_dict,
            )

            values_content: TextContent = res[0][0]
            value = values_content.split_ordered_list()
            value = [val.strip() for val in value]
            log_msg = (
                f"Values and Goals for {item['name']} ({i + 1}/{len(stakeholders)}):\n"
                + "\n".join([f"\t- {val}" for val in value])
                + f"\n{'*' * 5}"
            )
            logging.info(log_msg)
            item["values"] = value
            return item

    tasks = []
    for i, item in enumerate(stakeholders):
        tasks.append(
            asyncio.create_task(
                fetch_values(
                    i,
                    item,
                    deepcopy(message_list),
                    deepcopy(substitution_dict),
                    chatbot,
                )
            )
        )
    results = await asyncio.gather(*tasks)
    stakeholders = [res for res in results if res is not None]
    return stakeholders


# Identify potential losses from values using chatbot
async def identify_losses(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    values: list,
    dropout: float = 0.0,
):
    message_list = MessageList()

    # Instruction for converting a value into a loss
    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the description of a software system and a specific LLM-powered agents within that system, "
                "and considering a particular stakeholder associated with that agent, "
                "take a core abstract value that the stakeholder expects from the agent, "
                "and reverse it into the corresponding core abstract loss. "
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
                "Software System Description: \n"
                "{system_description}\n\n"
                "Agent Description: \n"
                "{agent_function}\n\n"
                "Stakeholder:\n"
                "{stakeholder}\n\n"
                "Values or goal:\n"
                "{value}\n\n"
                "Loss:\n"
            ),
        )
    )

    total_num = sum(len(item["values"]) if "values" in item else 0 for item in values)
    cnt_num = 0

    semaphore = asyncio.Semaphore(100)

    async def fetch_losses(
        i, j, id, item, value, message_list, substitution_dict, chatbot
    ):
        if dropout > 0 and random.random() < dropout and id > 5:
            return None
        async with semaphore:
            substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
            substitution_dict["value"] = value
            res, meta = await chatbot.completions(
                message_list,
                substitution_dict=substitution_dict,
            )
            loss_content: TextContent = res[0][0]
            loss = loss_content.text.strip()
            logging.info(f"({id}/{total_num}) Value: {value}\n\tLoss: {loss}")
            return (i, loss)

    tasks = []
    cnt_num = 0

    for i, item in enumerate(values):
        for j, val in enumerate(item.get("values", [])):
            cnt_num += 1
            tasks.append(
                asyncio.create_task(
                    fetch_losses(
                        i,
                        j,
                        cnt_num,
                        item,
                        val,
                        deepcopy(message_list),
                        deepcopy(substitution_dict),
                        chatbot,
                    )
                )
            )

    results = await asyncio.gather(*tasks)

    for res in results:
        if res is not None:
            i, loss = res
            if "losses" not in values[i]:
                values[i]["losses"] = []
            values[i]["losses"].append(loss)

    return values

    # # Loop through values and convert each into a potential loss
    # for i in range(len(values)):
    #     item = values[i]
    #     substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
    #     logging.info(f"Identifying losses for {item['name']}")
    #     if "values" not in item or len(item["values"]) == 0:
    #         logging.info(f"No values for {item['name']}, skipping...")
    #         continue
    #     for val in item["values"]:
    #         cnt_num += 1
    #         if dropout > 0 and random.random() < dropout:
    #             continue
    #         logging.info(f"({cnt_num}/{total_num}) Value: {val}")
    #         substitution_dict["value"] = val
    #         res, meta = chatbot.completions(
    #             message_list,
    #             substitution_dict=substitution_dict,
    #         )
    #         loss_content: TextContent = res[0][0]
    #         loss = loss_content.text.strip()
    #         logging.info(f"\tLoss: {loss}")
    #         if "losses" not in item:
    #             item["losses"] = []
    #         item["losses"].append(loss)
    #     logging.info(f"{'*' * 5}")
    #     values[i] = item
    # return values


# Identify hazards that could lead to each loss
async def identify_hazards(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    losses: list,
    dropout: float = 0.0,
):
    message_list = MessageList()

    # Instruction for identifying hazards
    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the description of a software system and a specific LLM-powered agents within that system, "
                "given a specific stakeholder, and given a specific loss that the stakeholder does not expect from the agent, "
                "focusing what the agent can do,"
                "identify and list potential actions of the agent that could directly lead to this loss under worst-case conditions. "
                "Provide concise, standalone descriptions of these actions. "
                "Do not include any cause, explanation, result, or solution to the action. "
                "Format your response as follows:\n"
                "1. Action 1\n"
                "2. Action 2\n"
                "3. Action 3\n"
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
                "Software System Description: \n"
                "{system_description}\n\n"
                "Agent Description: \n"
                "{agent_function}\n\n"
                "Stakeholder:\n"
                "{stakeholder}\n\n"
                "Loss:\n"
                "{loss}\n\n"
                "Actions of the Agent that could Lead to this Loss:\n"
            ),
        )
    )

    total_num = sum(len(item["losses"]) if "losses" in item else 0 for item in losses)
    logging.info(f"Total number of losses to process: {total_num}")
    cnt_num = 0

    semaphore = asyncio.Semaphore(100)

    async def fetch_hazards(
        i, j, id, item, loss, message_list, substitution_dict, chatbot
    ):
        if dropout > 0 and random.random() < dropout and id > 5:
            return None
        async with semaphore:
            substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
            substitution_dict["loss"] = loss
            res, meta = await chatbot.completions(
                message_list,
                substitution_dict=substitution_dict,
            )
            hazard_content: TextContent = res[0][0]
            hazard = hazard_content.split_ordered_list()
            hazard = [h.strip() for h in hazard]
            log_msg = (
                f"Hazards for {item['name']} - Loss: {loss} ({id}/{total_num}):\n"
                + "\n".join([f"\t- {h}" for h in hazard])
                + f"\n{'*' * 5}"
            )
            logging.info(log_msg)
            return (i, loss, hazard)

    tasks = []
    cnt_num = 0
    for i, item in enumerate(losses):
        for j, loss in enumerate(item.get("losses", [])):
            cnt_num += 1
            tasks.append(
                asyncio.create_task(
                    fetch_hazards(
                        i,
                        j,
                        cnt_num,
                        item,
                        loss,
                        deepcopy(message_list),
                        deepcopy(substitution_dict),
                        chatbot,
                    )
                )
            )
    results = await asyncio.gather(*tasks)
    for res in results:
        if res is not None:
            i, loss, hazard = res
            if "hazards" not in losses[i]:
                losses[i]["hazards"] = {}
            losses[i]["hazards"][loss] = hazard
    return losses

    # # Loop over each loss and collect hazards
    # for i in range(len(losses)):
    #     item = losses[i]
    #     substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
    #     logging.info(f"Identifying hazards for {item['name']}")
    #     item["hazards"] = {}
    #     if "losses" not in item or len(item["losses"]) == 0:
    #         logging.info(f"No losses for {item['name']}, skipping...")
    #         continue
    #     for j in range(len(item["losses"])):
    #         cnt_num += 1
    #         if dropout > 0 and random.random() < dropout:
    #             continue
    #         logging.info(f"({cnt_num}/{total_num}) Loss: {item['losses'][j]}")
    #         loss = item["losses"][j]
    #         substitution_dict["loss"] = loss
    #         res, meta = chatbot.completions(
    #             message_list,
    #             substitution_dict=substitution_dict,
    #         )
    #         hazard_content: TextContent = res[0][0]
    #         hazard = hazard_content.split_ordered_list()
    #         hazard = [h.strip() for h in hazard]
    #         logging.info(f"Hazards for {loss}:")
    #         for h in hazard:
    #             logging.info(f"\t- {h}")
    #         logging.info(f"{'*' * 5}")
    #         item["hazards"][loss] = hazard
    #     losses[i] = item
    # return losses


# Consolidate all hazard statements by clustering and summarizing them
async def consolidate_hazards(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    hazards_comprehensive: list,
    n_clusters=20,
    random_sample_size=1.0,
):
    hazard_list = []
    for item in hazards_comprehensive:
        for loss, hazards in item["hazards"].items():
            hazard_list.extend(hazards)
    logging.info(f"Total number of hazards before consolidation: {len(hazard_list)}")

    random_sample_size = min(1.0, random_sample_size)
    hazard_list = random.sample(hazard_list, int(len(hazard_list) * random_sample_size))

    logging.info(
        f"Number of hazards after random sampling (size={random_sample_size}): {len(hazard_list)}"
    )

    consolidated_hazards = await consolidate_hazard_list(
        chatbot, substitution_dict, hazard_list, n_clusters=n_clusters
    )
    logging.info(
        f"Total number of hazards after consolidation: {len(consolidated_hazards)}"
    )
    return consolidated_hazards


# Break down hazard list into segments, consolidate each segment
async def divide_and_consolidate(
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
        segment = await consolidate_hazard_list(
            chatbot, substitution_dict, segment, n_clusters=n_clusters
        )
        res.extend(segment)
    return res


# Core function to group similar hazards using embeddings and clustering
async def consolidate_hazard_list(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    hazard_list: list,
    n_clusters=20,
):
    if n_clusters != 1:
        logging.info("Getting embeddings for hazards...")

        with ThreadPoolExecutor(max_workers=10) as executor:
            embeddings = list(executor.map(get_embedding, hazard_list))

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
                "Based on the description of a system with agentic components, review the list of potential risks and identify those that have close meanings. "
                "Merge similar risks into a single entry. "
                "Each entry should be a standalone description. "
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
                "Software System Description: \n"
                "{system_description}\n\n"
                "Agent Description: \n"
                "{agent_function}\n\n"
                "State or Condition List:\n"
                "{hazard_list}\n\n"
                "Merged States or Conditions:\n"
            ),
        )
    )

    semaphore = asyncio.Semaphore(20)

    async def fetch_consolidated_hazards(
        i,
        cluster,
        hazards,
        message_list,
        substitution_dict,
        chatbot: ChatCompletionEndPoint,
    ):
        async with semaphore:
            substitution_dict["hazard_list"] = "\n".join(
                [f"- {hazard}" for hazard in hazards]
            )
            res, meta = await chatbot.completions(
                message_list,
                substitution_dict=substitution_dict,
            )
            consolidated_hazards_content: TextContent = res[0][0]
            consolidated_hazards = consolidated_hazards_content.split_ordered_list()
            consolidated_hazards = [h.strip() for h in consolidated_hazards]
            logging.info(
                f"Consolidated Hazards for Cluster {cluster} ({i+1}/{len(hazard_clusters)}):"
            )
            for h in consolidated_hazards:
                logging.info(f"\t- {h}")
            logging.info(f"{'*' * 5}")
            return consolidated_hazards

    tasks = []
    for i, (cluster, hazards) in enumerate(hazard_clusters.items()):
        tasks.append(
            asyncio.create_task(
                fetch_consolidated_hazards(
                    i,
                    cluster,
                    hazards,
                    deepcopy(message_list),
                    deepcopy(substitution_dict),
                    chatbot,
                )
            )
        )
    results = await asyncio.gather(*tasks)
    res_list = []
    for consolidated_hazards in results:
        res_list.extend(consolidated_hazards)
    return res_list
