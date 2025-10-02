from OpenAIChatHelper import ChatCompletionEndPoint
from OpenAIChatHelper.message import (
    SubstitutionDict,
    MessageList,
    DevSysUserMessage,
    TextContent,
)

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


async def filter_value(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    value_list: list,
    dropout: float = 0.0,
):
    message_list = MessageList()

    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the description of a software system and a specific function within that system that is enabled by LLM-powered agents, "
                "and considering a particular stakeholder associated with that function, "
                "given a core abstract value that the stakeholder expects from the function, "
                "evaluate the importance of this value from the perspective of the general social good.\n"
                "Classify the importance as:\n"
                "- High: The value is very important and should always be enforced.\n"
                "- Medium: The value is desirable; lacking it would have noticeable but not severe consequences.\n"
                "- Low: The value is good to have, but not essential.\n"
                "Format your response as follows:\n"
                "Importance: [High/Medium/Low]\n"
                "A short reason why the value is important\n"
            ),
        )
    )

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
                "Value:\n"
                "{value}\n\n"
                "Response:\n"
            ),
        )
    )

    total_value_cnt = sum(len(item.get("values", [])) for item in value_list)
    logging.info(f"Total values to evaluate: {total_value_cnt}")

    semaphore = asyncio.Semaphore(100)

    async def evaluate_value(
        id, i, item, value, substitution_dict, message_list, chatbot
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
            importance_content: TextContent = res[0][0]
            importance_text = importance_content.text.strip()
            log_text = f"({id}/{total_value_cnt}) Stakeholder: {item['name']}\nValue: {value}\nImportance: {importance_text}"
            logging.info(log_text)
            first_answer_line = importance_text.split("\n")[0].strip().lower()
            if "high" in first_answer_line:
                return (i, value)
            return None

    tasks = []
    cnt = 0

    for i, item in enumerate(value_list):
        for j, value in enumerate(item.get("values", [])):
            cnt += 1
            tasks.append(
                asyncio.create_task(
                    evaluate_value(
                        cnt,
                        i,
                        item,
                        value,
                        deepcopy(substitution_dict),
                        deepcopy(message_list),
                        chatbot,
                    )
                )
            )
    results = await asyncio.gather(*tasks)
    for res in results:
        if res is not None:
            i, value = res
            if "filtered_values" not in value_list[i]:
                value_list[i]["filtered_values"] = []
            value_list[i]["filtered_values"].append(value)
    for i in range(len(value_list)):
        if "filtered_values" in value_list[i]:
            value_list[i]["original_values"] = value_list[i]["values"]
            value_list[i]["values"] = value_list[i]["filtered_values"]
            del value_list[i]["filtered_values"]
        else:
            value_list[i]["original_values"] = value_list[i]["values"]
            value_list[i]["values"] = []
    return value_list

    # for i, item in enumerate(value_list):
    #     name = item["name"]
    #     description = item["description"]
    #     substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
    #     if "values" not in item or len(item["values"]) == 0:
    #         logging.info(f"No values for {item['name']}, skipping...")
    #         continue
    #     values = item["values"]
    #     filtered_values = []
    #     for value in values:
    #         cnt += 1
    #         logging.info(f"Evaluating value {cnt}/{total_value_cnt}")
    #         if dropout > 0 and random.random() < dropout:
    #             continue
    #         substitution_dict["value"] = value
    #         res, meta = chatbot.completions(
    #             message_list,
    #             substitution_dict=substitution_dict,
    #         )
    #         importance_content: TextContent = res[0][0]
    #         importance_text = importance_content.text.strip()
    #         logging.info(f"Value: {value}")
    #         logging.info(f"Importance: {importance_text}")
    #         first_answer_line = importance_text.split("\n")[0].strip().lower()
    #         if "high" in first_answer_line:
    #             filtered_values.append(value)
    #     value_list[i]["original_values"] = value_list[i]["values"]
    #     value_list[i]["values"] = filtered_values
    # return value_list


async def filter_loss(
    chatbot: ChatCompletionEndPoint,
    substitution_dict: SubstitutionDict,
    loss_list: list,
    dropout: float = 0.0,
):
    message_list = MessageList()

    message_list.add_message(
        DevSysUserMessage(
            "system",
            TextContent(
                "Based on the description of a software system and a specific function within that system that is enabled by LLM-powered agents, "
                "and considering a particular stakeholder associated with that function, "
                "given a core abstract loss that the stakeholder may face from the function, "
                "evaluate the severity of this loss from the perspective of the general social good.\n"
                "Classify the severity as:\n"
                "- High: The loss is very severe and should always be mitigated.\n"
                "- Medium: The loss is noticeable but not severe.\n"
                "- Low: The loss is undesirable, but minor and not essential to mitigate.\n"
                "Format your response as follows:\n"
                "Severity: [High/Medium/Low]\n"
                "A short reason why the loss is severe\n"
            ),
        )
    )

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
                "Response:\n"
            ),
        )
    )

    total_loss_cnt = sum(len(item.get("losses", [])) for item in loss_list)
    logging.info(f"Total losses to evaluate: {total_loss_cnt}")

    semaphore = asyncio.Semaphore(100)

    async def evaluate_loss(
        id, i, item, loss, substitution_dict, message_list, chatbot
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
            severity_content: TextContent = res[0][0]
            severity_text = severity_content.text.strip()
            log_text = f"({id}/{total_loss_cnt}) Stakeholder: {item['name']}\nLoss: {loss}\nSeverity: {severity_text}"
            logging.info(log_text)
            first_answer_line = severity_text.split("\n")[0].strip().lower()
            if "high" in first_answer_line:
                return (i, loss)
            return None

    cnt = 0

    tasks = []

    for i, item in enumerate(loss_list):
        for j, loss in enumerate(item.get("losses", [])):
            cnt += 1
            tasks.append(
                asyncio.create_task(
                    evaluate_loss(
                        cnt,
                        i,
                        item,
                        loss,
                        deepcopy(substitution_dict),
                        deepcopy(message_list),
                        chatbot,
                    )
                )
            )
    results = await asyncio.gather(*tasks)
    for res in results:
        if res is not None:
            i, loss = res
            if "filtered_losses" not in loss_list[i]:
                loss_list[i]["filtered_losses"] = []
            loss_list[i]["filtered_losses"].append(loss)
    for i in range(len(loss_list)):
        if "filtered_losses" in loss_list[i]:
            loss_list[i]["original_losses"] = loss_list[i]["losses"]
            loss_list[i]["losses"] = loss_list[i]["filtered_losses"]
            del loss_list[i]["filtered_losses"]
        else:
            loss_list[i]["original_losses"] = loss_list[i]["losses"]
            loss_list[i]["losses"] = []
    return loss_list

    # for i, item in enumerate(loss_list):
    #     name = item["name"]
    #     description = item["description"]
    #     losses = item["losses"]
    #     substitution_dict["stakeholder"] = f"{item['name']} - {item['description']}"
    #     if "losses" not in item or len(item["losses"]) == 0:
    #         logging.info(f"No losses for {item['name']}, skipping...")
    #         continue
    #     filtered_losses = []
    #     for loss in losses:
    #         cnt += 1
    #         logging.info(f"Evaluating loss {cnt}/{total_loss_cnt}")
    #         if dropout > 0 and random.random() < dropout:
    #             continue
    #         substitution_dict["loss"] = loss
    #         res, meta = chatbot.completions(
    #             message_list,
    #             substitution_dict=substitution_dict,
    #         )
    #         severity_content: TextContent = res[0][0]
    #         severity_text = severity_content.text.strip()
    #         logging.info(f"Loss: {loss}")
    #         logging.info(f"Severity: {severity_text}")
    #         first_answer_line = severity_text.split("\n")[0].strip().lower()
    #         if "high" in first_answer_line:
    #             filtered_losses.append(loss)
    #     loss_list[i]["original_losses"] = loss_list[i]["losses"]
    #     loss_list[i]["losses"] = filtered_losses
    # return loss_list
