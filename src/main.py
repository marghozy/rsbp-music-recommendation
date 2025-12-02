from rule_engine import Rule, RuleEngine
from cbr import Case, CaseBase
from dataset_loader import load_dataset
import numpy as np

# ----------- RULE BASE -----------
def build_rules():
    return [
        Rule("R1", {"activity": "workout"}, {"energy": "high", "tempo": "fast"}, priority=10, description="Workout → high energy, fast tempo"),
        Rule("R2", {"mood": "happy"}, {"valence": "high"}, priority=7, description="Happy mood → high valence"),
        Rule("R3", {"activity": "study"}, {"acousticness": "high"}, priority=6, description="Study → calm & acoustic"),
        Rule("R4", {"mood": "sad"}, {"valence": "low"}, priority=5, description="Sad → low valence")
    ]

# ----------- MAIN EXECUTION ----------
def main():
    print("Loading dataset...")
    df = load_dataset("data/dataset.csv")

    rules = build_rules()
    engine = RuleEngine(rules)

    # Example user input (can later connect to UI)
    user_context = {
        "activity": "study",
        "mood": "sad",
        "time_of_day": "morning"
    }

    print("\nUser context:", user_context)

    # -------- RULE-BASED INFERENCE --------
    inferred_actions, trace = engine.infer(user_context)
    print("\nRule inference:", inferred_actions)

    filtered = df.copy()
    for feature, value in inferred_actions.items():
        col = f"{feature}_cat"
        if col in filtered.columns:
            filtered = filtered[filtered[col] == value]
        else:
            print(f"⚠ No dataset feature mapped for {feature}")

    print(f"\nRule-based candidates: {len(filtered)} tracks")

    # -------- CASE-BASED RETRIEVAL --------
    case_base = CaseBase()

    # Load previous cases if exist
    try:
        case_base.load("data/casebase.json")
        print(f"Loaded {len(case_base.cases)} existing cases.")
    except:
        print("No previous case base found, starting fresh.")

    # Convert rule inference into numeric similarity vector
    # For now using neutral guessed values (can refine later)
    query_vector = np.array([0.9, 0.7, 0.8, 0.3, 0.75])  

    retrieved = case_base.retrieve(query_vector.tolist(), k=3)
    print("\nCBR retrieved:", [(c.case_id, dist) for c, dist in retrieved])

    # -------- HYBRID SCORING --------
    W_rule = 0.6
    W_cbr = 0.4

    scores = {}
    for _, row in filtered.head(30).iterrows():  # evaluate top slice
        rule_score = 1.0
        
        # compute similarity score from retrieved case
        sim_scores = []
        for case, dist in retrieved:
            sim = 1 / (1 + dist)
            sim_scores.append(sim)

        case_score = max(sim_scores) if sim_scores else 0
        final_score = (W_rule * rule_score) + (W_cbr * case_score)

        scores[row.track_name] = final_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("\n🎵 Final Ranked Recommendations:\n")
    for song, sc in ranked[:10]:
        print(f"- {song} (score={sc:.3f})")

    print("\n📌 Explanation:")
    for step in trace:
        print(" •", step)

    if retrieved:
        print(f" • Best matching case: {retrieved[0][0].case_id} (distance={retrieved[0][1]:.3f})")

    # -------- CASE RETENTION --------
    feedback = input("\nDid you like these recommendations? (yes/no OR 1-5 rating): ")
    try:
        feedback_value = float(feedback)
    except ValueError:
        feedback_value = 0.0  # default or handle non-numeric input

    case_base.add_case(Case(
        case_id=f"case_{len(case_base.cases)+1}",
        context=user_context,
        features=query_vector.tolist(),
        recommended=[{"track_name": x[0]} for x in ranked[:5]],  
        feedback=feedback_value
    ))

    case_base.save("data/casebase.json")
    print("\n💾 Case saved! System has now learned from this session.")

if __name__ == "__main__":
    main()