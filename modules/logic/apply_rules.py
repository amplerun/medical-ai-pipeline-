import json

class RuleEngine:
    def __init__(self, path="rules/rules.json"):
        with open(path, "r") as f:
            self.rules = json.load(f)

    def run_all(self, diagnosis=[], labs={}, indicators=[]):
        triggered = []
        for rule in self.rules.get("conflicting_diagnoses", []):
            expr = rule["rule"]
            try:
                if eval(expr.replace("AND", "and").replace("OR", "or")):
                    triggered.append(rule["action"])
            except:
                pass
        return triggered
