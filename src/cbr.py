"""
Case-Based Reasoning (CBR) module.
Simple case representation + KD-tree indexing for numeric profiles.
Each case stores: context (dict), features (numeric vector), recommended_items (list), feedback (score)
"""

import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.neighbors import KDTree

class Case:
    def __init__(self, case_id: str, context: Dict[str, Any], features: List[float], recommended: List[Dict[str,Any]], feedback: float = 1.0):
        self.case_id = case_id
        self.context = context
        self.features = np.array(features, dtype=float)
        self.recommended = recommended
        self.feedback = feedback

    def to_dict(self):
        return {
            "case_id": self.case_id,
            "context": self.context,
            "features": self.features.tolist(),
            "recommended": self.recommended,
            "feedback": self.feedback
        }

    @staticmethod
    def from_dict(d):
        return Case(d["case_id"], d["context"], d["features"], d["recommended"], d.get("feedback",1.0))


class CaseBase:
    def __init__(self, cases: Optional[List[Case]] = None):
        self.cases = cases or []
        self._build_index()

    def add_case(self, case: Case):
        self.cases.append(case)
        self._build_index()

    def _build_index(self):
        if len(self.cases) == 0:
            self.kdt = None
            self.matrix = None
            return
        self.matrix = np.vstack([c.features for c in self.cases])
        self.kdt = KDTree(self.matrix, leaf_size=10)

    def retrieve(self, query_features: List[float], k: int = 5) -> List[Tuple[Case, float]]:
        if self.kdt is None:
            return []
        q = np.array(query_features, dtype=float).reshape(1, -1)
        dist, idx = self.kdt.query(q, k=min(k, len(self.cases)))
        results = []
        for d, i in zip(dist[0], idx[0]):
            results.append((self.cases[i], float(d)))
        return results

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump([c.to_dict() for c in self.cases], f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            arr = json.load(f)
        self.cases = [Case.from_dict(d) for d in arr]
        self._build_index()