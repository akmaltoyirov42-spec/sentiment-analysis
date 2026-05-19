import streamlit as st
from src.predict import predict

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 Sentiment Analyzer")
st.write("Type a movie review and the model will tell you if it's positive or negative.")

text = st.text_area("Review", placeholder="The movie was...", height=150)

if st.button("Analyze", use_container_width=True) and text.strip():
    try:
        result = predict(text)

        color = "#27ae60" if result["sentiment"] == "positive" else "#e74c3c"
        st.markdown(f"### <span style='color:{color}'>{result['sentiment'].upper()}</span>", unsafe_allow_html=True)
        st.markdown(f"Confidence: **{result['confidence']*100:.1f}%**")

        st.divider()
        col1, col2 = st.columns(2)
        col1.metric("Positive", f"{result['positive_prob']*100:.1f}%")
        col2.metric("Negative", f"{result['negative_prob']*100:.1f}%")

    except FileNotFoundError:
        st.error("No trained model found. Run `python -m src.train` first.")
