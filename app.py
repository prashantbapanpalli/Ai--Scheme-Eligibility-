"""Streamlit app: tell it about yourself, see which government schemes fit.

Run:  streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from scheme_finder import Profile, SchemeAssistant
from scheme_finder.assistant import llm_configured
from scheme_finder.eligibility import LIKELY, MAYBE, NO

DATA = Path(__file__).parent / "data" / "gov_myscheme_data.csv"
BADGE = {LIKELY: ("🟢", "Likely eligible"), MAYBE: ("🟡", "Check details"), NO: ("🔴", "Probably not eligible")}

st.set_page_config(page_title="India Scheme Finder", page_icon="🇮🇳", layout="wide")


@st.cache_resource(show_spinner="Loading 2,066 schemes…")
def load() -> SchemeAssistant:
    return SchemeAssistant(pd.read_csv(DATA))


A = load()
states = sorted(A.df.loc[A.df["Level"] != "Central Government", "State / UT / Ministry"].unique())

# ------------------------------------------------------------------ profile
with st.sidebar:
    st.header("About you")
    st.caption("Everything is optional. Nothing is stored or sent anywhere"
               + (" except the LLM you configured." if llm_configured() else "."))
    state = st.selectbox("State / UT", ["(any)"] + states, index=(states.index("Karnataka") + 1))
    age = st.number_input("Age", min_value=0, max_value=110, value=None, step=1, placeholder="e.g. 21")
    gender = st.radio("Gender", ["Prefer not to say", "Female", "Male"], horizontal=True)
    income_lakh = st.number_input("Annual family income (₹ lakh)", min_value=0.0, max_value=100.0, value=None,
                                  step=0.5, placeholder="e.g. 2.5")
    category = st.selectbox("Category", ["Prefer not to say", "General", "SC", "ST", "OBC"])
    minority = st.checkbox("Minority community")
    bpl = st.checkbox("BPL card holder")
    disability = st.checkbox("Person with disability")
    occupation = st.multiselect("You are a…", ["student", "farmer", "worker", "entrepreneur", "senior", "unemployed"])
    show_no = st.toggle("Also show schemes I'm probably not eligible for", value=False)

profile = Profile(
    state=None if state == "(any)" else state,
    age=int(age) if age is not None else None,
    gender={"Female": "female", "Male": "male"}.get(gender),
    annual_income=income_lakh * 100_000 if income_lakh is not None else None,
    category={"General": "general", "SC": "sc", "ST": "st", "OBC": "obc"}.get(category),
    minority=minority, bpl=bpl, disability=disability, occupation=occupation,
)

# ------------------------------------------------------------------ search
st.title("India Scheme Finder")
st.write("Find central and state government schemes that fit you, with the reason for each match. "
         "Built on 2,066 schemes from the official **myScheme** portal.")

question = st.text_input("What do you need help with?", placeholder="e.g. scholarship for engineering, pension, loan to start a shop")
if st.button("Find schemes", type="primary") or question:
    st.session_state.matches = A.find(profile, question, k=12, include_unlikely=show_no)
    st.session_state.question = question

matches = st.session_state.get("matches")
if matches is not None:
    st.subheader("Where to start")
    st.markdown(A.summarise(profile, matches, st.session_state.get("question", "")))
    counts = pd.Series([m.status for m in matches]).value_counts()
    st.caption(" · ".join(f"{BADGE[s][0]} {n} {BADGE[s][1].lower()}" for s, n in counts.items()))

    for n, m in enumerate(matches):
        icon, label = BADGE[m.status]
        with st.container(border=True):
            st.markdown(f"**{m.name}**  \n{icon} {label} — {'; '.join(m.reasons)}")
            st.caption(f"{'Central' if m.level == 'Central Government' else 'State'} · {m.where}")
            st.write(m.benefit)
            with st.expander("Details and questions"):
                row = A.df.loc[m.idx]
                tab1, tab2, tab3, tab4 = st.tabs(["Eligibility", "Benefits", "How to apply", "Documents"])
                tab1.write(row["Eligibility Criteria"])
                tab2.write(row["Benefits"])
                tab3.write(row["Application Process"])
                tab4.write(row["Documents Required"])
                q = st.text_input("Ask about this scheme", key=f"q{m.idx}", placeholder="e.g. what documents do I need?")
                if q:
                    res = A.ask(m.idx, q)
                    st.info(res["answer"])
                    with st.popover("Sources"):
                        for lab, txt in res["sources"]:
                            st.markdown(f"**{lab}** — {txt}")
                st.link_button("Official page", m.link)

st.divider()
st.caption("Eligibility is read from the scheme text automatically and can be wrong — always confirm on the "
           "official page before applying. Data: myScheme (Government of India) via the "
           "Aryan-Pardeshi/gov-myscheme-dataset repository, Apache-2.0. Not an official government service.")
