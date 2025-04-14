import json

class RuleEngine:
    def __init__(self, rules_path="modules/logic/rules.json"):
        with open(rules_path, "r") as f:
            self.rules = json.load(f)

    def run_all(self, diagnosis, labs, indicators):
        triggered = {"conflicts": [], "labs": [], "requirements": []}

        for rule in self.rules["conflicting_diagnoses"]:
            if all(term in diagnosis for term in rule["rule"].split(" AND ")):
                triggered["conflicts"].append(rule["rule"])

        for rule in self.rules["lab_rules"]:
            lab_val = labs.get(rule["lab"])
            if lab_val and lab_val >= rule["min"]:
                triggered["labs"].append(f"{rule['lab']} – {rule['message']}")

        for rule in self.rules["requirements"]:
            if all(r in indicators + diagnosis for r in rule["requires"]):
                triggered["requirements"].append(rule["message"])

        return triggered

