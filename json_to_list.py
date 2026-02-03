import json

HAZARD_JSON = "hazards.json"

with open(HAZARD_JSON, "r") as f:
    hazard_json = json.load(f)

res = []

for item in hazard_json:
    if "hazards" not in item:
        continue
    for loss in item["hazards"]:
        for hazard in item["hazards"][loss]:
            # res.append(
            #     {
            #         "stakeholder": item["name"],
            #         "stakeholder_description": item["description"],
            #         "loss": loss,
            #         "hazard": hazard,
            #     }
            # )
            res.append(hazard)

with open("hazard_list.json", "w") as f:
    json.dump(res, f, indent=4)

# random pick 10 and print
import random

sampled = random.sample(res, 10)
for s in sampled:
    print(s)
