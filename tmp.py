import json

path = "/Users/yhong3/Documents/Research/Software Security/working_repo/Hazard-Finder/hazard_api_check_results.json"

with open(path, "r") as f:
    hazard_list = json.load(f)

# count true/false for human_annotated_api_check and api_check_possible, if the key exists

cnt_true_human = 0
cnt_false_human = 0
cnt_true_api = 0
cnt_false_api = 0
human_api_match = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

for hazard in hazard_list:
    if "human_annotated_api_check" in hazard:
        if hazard["human_annotated_api_check"]:
            cnt_true_human += 1
        else:
            cnt_false_human += 1
    if "api_check_possible" in hazard:
        if hazard["api_check_possible"]:
            cnt_true_api += 1
        else:
            cnt_false_api += 1
    if "human_annotated_api_check" in hazard and "api_check_possible" in hazard:
        if hazard["human_annotated_api_check"] and hazard["api_check_possible"]:
            human_api_match["TP"] += 1
        elif (
            not hazard["human_annotated_api_check"] and not hazard["api_check_possible"]
        ):
            human_api_match["TN"] += 1
        elif not hazard["human_annotated_api_check"] and hazard["api_check_possible"]:
            human_api_match["FP"] += 1
        elif hazard["human_annotated_api_check"] and not hazard["api_check_possible"]:
            human_api_match["FN"] += 1

print(
    f"Human annotated API check - True: {cnt_true_human} ({cnt_true_human/(cnt_true_human+cnt_false_human)*100:.2f}%), False: {cnt_false_human} ({cnt_false_human/(cnt_true_human+cnt_false_human)*100:.2f}%)"
)
print(
    f"API check possible - True: {cnt_true_api} ({cnt_true_api/(cnt_true_api+cnt_false_api)*100:.2f}%), False: {cnt_false_api} ({cnt_false_api/(cnt_true_api+cnt_false_api)*100:.2f}%)"
)
print(
    f"Human vs API check match - TP: {human_api_match['TP']}, TN: {human_api_match['TN']}, FP: {human_api_match['FP']}, FN: {human_api_match['FN']}"
)
total_hazards = len(hazard_list)

for index, hazard in enumerate(hazard_list):
    if "human_annotated_api_check" in hazard:
        continue
    print("=====================================")
    print(f"Hazard {index}/{total_hazards}:")
    for key, value in hazard.items():
        print(f"Hazard {index} - {key}: {value}")
    while True:
        user_input = input("Is API check possible? (Y/N): ")
        if user_input.lower() == "y":
            hazard["human_annotated_api_check"] = True
            break
        elif user_input.lower() == "n":
            hazard["human_annotated_api_check"] = False
            break
        else:
            print("Invalid input. Please enter Y or N.")
    user_note = input("Add human notes (or press Enter to skip): ")
    hazard["human_notes"] = user_note
    with open(path, "w") as f:
        json.dump(hazard_list, f, indent=4)
