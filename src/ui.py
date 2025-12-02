import streamlit as st
import pandas as pd
from rule_engine import Rule, RuleEngine
from cbr import Case, CaseBase
from dataset_loader import load_dataset
import os

CASE_PATH = "data/cases.json"

# ---------- UI Theme & Layout ----------
st.set_page_config(page_title="AI Music Recommender", page_icon="🎧", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fc; }
    h1, h2, h3, h4 { font-family: 'Segoe UI', sans-serif; }
    .song-box { padding: 10px; margin-bottom: 10px; border-radius: 10px; background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# ---------- RULES ----------
def build_rules():
    return [
        Rule("R1", {"activity": "workout"}, {"energy": "high", "tempo": "fast"}, priority=10),
        Rule("R2", {"mood": "happy"}, {"valence": "high"}, priority=7),
        Rule("R3", {"activity": "study"}, {"acousticness": "high"}, priority=6),
        Rule("R4", {"mood": "sad"}, {"valence": "low"}, priority=5),
    ]


# ---------- CASE LOADER ----------
def load_case_base():
    case_base = CaseBase()
    try:
        case_base.load(CASE_PATH)
        return case_base, True
    except:
        return CaseBase(), False


# ---------- INPUT UI ----------
st.title("🎵 Intelligent Music Recommendation System")

st.sidebar.title("⚙ Preferences")
activity = st.sidebar.selectbox("Activity", ["Workout", "Study", "Relax", "Commute"])
mood = st.sidebar.selectbox("Mood", ["Happy", "Neutral", "Sad"])
time_of_day = st.sidebar.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

st.header("🎧 Personalized Music Suggestions Based on You")


# ---------- PROCESS REQUEST ----------
if st.button("🔍 Get Recommendations"):
    with st.spinner("Thinking..."):
        
        df = load_dataset("data/dataset.csv")
        user_context = {"activity": activity, "mood": mood, "time_of_day": time_of_day}

        # Rule reasoning
        engine = RuleEngine(build_rules())
        inferred, trace = engine.infer(user_context)

        # Filter dataset based on rules
        filtered = df.copy()
        for feature, value in inferred.items():
            if f"{feature}_cat" in filtered.columns:
                filtered = filtered[filtered[f"{feature}_cat"] == value]

        st.success("Results Ready 🎶")

        case_base, exists = load_case_base()

        # Select first matching vector
        query_vec = filtered.iloc[0]["vector"] if len(filtered) else [0,0,0,0,0]
        query_vec = [float(x) for x in query_vec]
        retrieved = case_base.retrieve(query_vec, k=1)

        dist = None
        case = None
        if retrieved:
            case, dist = retrieved[0]
            final_score = max(1 - dist, 0.3)
        else:
            final_score = 0.6

        top_songs = filtered.head(10)

        # ---------- DISPLAY RESULTS ----------
        st.subheader("🔥 Top Recommended Tracks")

        for idx, row in enumerate(top_songs.reset_index().itertuples(), 1):
            st.markdown(f"""
            <div class="song-box">
            <b>{idx}. {row.track_name}</b><br>
            <i>{row.artists}</i> — Score: <b>{round(final_score,3)}</b>
            </div>
            """, unsafe_allow_html=True)

            # Spotify Player Embed
            embed_url = f"https://open.spotify.com/embed/track/{row.track_id}"
            st.markdown(f'<iframe src="{embed_url}" width="100%" height="80" frameborder="0"></iframe>', unsafe_allow_html=True)
        # ---------- EXPLANATION ----------
        with st.expander("🧠 Show Reasoning"):
            st.write("### Fired Rules:")
            for t in trace:
                st.write("•", t)
            if retrieved and dist is not None and case is not None:
                st.write(f"\n### Case Used: `{case.case_id}` (distance={dist:.3f})")
                st.write(f"\n### Case Used: `{case.case_id}` (distance={dist:.3f})")
                st.write(f"\n### Case Used: `{case.case_id}` (distance={dist:.3f})")


        # ---------- FEEDBACK + METRICS ----------
        rating = st.slider("Rate these recommendations (1 = bad, 5 = great):", 1, 5)

        if st.button("💾 Submit Feedback"):
            
            # Evaluation Metrics
            precision_at_k = 1 if rating >= 4 else 0
            mrr = 1  # since ranking is deterministic

            new_case = Case(
                case_id=f"case_{len(case_base.cases)+1}",
                context=user_context,
                features=list(filtered.iloc[0]["vector"]),
                recommended=[{str(k): v for k, v in row.items()} for row in top_songs.to_dict(orient="records")],
                feedback=rating / 5
            )

            case_base.add_case(new_case)
            os.makedirs("data", exist_ok=True)
            case_base.save(CASE_PATH)

            st.success("👍 Thank you — your feedback helped the system learn!")


        # ---------- SYSTEM PERFORMANCE ----------
        with st.expander("📈 System Performance Over Time"):
            all_scores = [c.feedback for c in case_base.cases] if case_base.cases else []

            if len(all_scores) > 0:
                st.line_chart(all_scores)
                st.write(f"Avg User Satisfaction: **{round(sum(all_scores)/len(all_scores),2)}**")
            else:
                st.info("No feedback stored yet.")