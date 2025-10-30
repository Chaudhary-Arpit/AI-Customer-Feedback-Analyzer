import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import os

st.set_page_config(page_title="Customer Feedback Analyzer", layout="wide")
st.title("🧠 Intelligent Customer Feedback Analysis System")

st.markdown("Upload a CSV with columns: `id`, `feedback` or use the sample included.")

uploaded_file = st.file_uploader("Upload feedback CSV", type=['csv'])
if uploaded_file is None:
    if st.button("Use sample_feedback.csv"):
        uploaded_file = "sample_feedback.csv"

if uploaded_file is not None:
    if isinstance(uploaded_file, str):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # basic preprocessing
    import re
    def clean_text(text):
        if pd.isna(text): 
            return ""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text)).lower()
        return text

    df['cleaned_feedback'] = df['feedback'].apply(clean_text)

    st.subheader("Cleaned Feedback (sample)")
    st.dataframe(df[['id','cleaned_feedback']].head())

    st.subheader("Sentiment Analysis (using a small pipeline)")
    try:
        sentiment = pipeline("sentiment-analysis")
        df['sentiment'] = df['feedback'].apply(lambda x: sentiment(str(x)[:512])[0]['label'])
    except Exception as e:
        st.warning("Could not load transformer pipeline in this environment. Displaying a simple rule-based sentiment instead.")
        def rule_sent(x):
            txt = str(x).lower()
            if any(w in txt for w in ['terrible','bad','poor','frustrat','missing','damag','crash']):
                return 'NEGATIVE'
            if any(w in txt for w in ['excellent','amazing','great','satisfied','helpful','fast']):
                return 'POSITIVE'
            return 'NEUTRAL'
        df['sentiment'] = df['cleaned_feedback'].apply(rule_sent)

    st.write(df[['id','feedback','sentiment']].head(20))

    st.subheader("Sentiment Distribution")
    dist = df['sentiment'].value_counts()
    st.bar_chart(dist)

    st.subheader("Short Summaries (rule-of-thumb demo)")
    # simple extractive short summary: first 10 words
    df['short_summary'] = df['cleaned_feedback'].apply(lambda x: ' '.join(x.split()[:10]))
    st.dataframe(df[['id','short_summary']].head())

    st.download_button("Download analyzed CSV", df.to_csv(index=False), file_name="analyzed_feedback.csv")
