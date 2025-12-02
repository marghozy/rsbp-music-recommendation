"""
Simple forward-chaining rule engine for music recommendation.
Rules are defined as Python dicts with conditions over discretized features.
Each rule has: id, conditions (dict), actions (dict), priority (int), description
"""

from typing import Dict, List, Any, Tuple, Optional


class Rule:
    def __init__(self, rule_id: str, conditions: Dict[str, Any], actions: Dict[str, Any], priority: int = 0, description: str = ""):
        self.id = rule_id
        self.conditions = conditions  # e.g., {"activity": "workout", "time_of_day": "morning"}
        self.actions = actions        # e.g., {"energy": "high", "tempo": "fast"}
        self.priority = priority
        self.description = description

    def matches(self, facts: Dict[str, Any]) -> bool:
        # all conditions must be satisfied by facts
        for k, v in self.conditions.items():
            if k not in facts:
                return False
            # allow list of values or single value
            if isinstance(v, list):
                if facts[k] not in v:
                    return False
            else:
                if facts[k] != v:
                    return False
        return True


class RuleEngine:
    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules = rules or []

    def add_rule(self, rule: Rule):
        self.rules.append(rule)
        # keep rules sorted by priority desc
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def infer(self, facts: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Apply forward chaining: return inferred actions and a trace of fired rules.
        """
        inferred = {}
        trace = []

        for rule in self.rules:
            if rule.matches(facts):
                # apply actions
                for k, v in rule.actions.items():
                    # if action already set, skip or override based on priority
                    if k in inferred:
                        # skip to keep earlier higher-priority rule; could be changed
                        continue
                    inferred[k] = v
                trace.append(f"Rule fired: {rule.id} - {rule.description}")

        return inferred, trace