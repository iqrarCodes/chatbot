import os
import streamlit as st
from groq import Groq
import spacy
from textblob import TextBlob

# -------------------------------------------------------------------
# 1. Load API Key – use st.secrets on Cloud, fallback to .env locally
# -------------------------------------------------------------------
try:
    # For Streamlit Cloud deployment
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    # For local development (requires python-dotenv installed)
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=api_key)

# -------------------------------------------------------------------
# 2. Load spaCy model with caching (essential for cloud performance)
# -------------------------------------------------------------------
@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# -------------------------------------------------------------------
# 3. Page config & title
# -------------------------------------------------------------------
st.set_page_config(page_title="Groq NLP Agent", page_icon="🧠")
st.title("🧠 Groq NLP‑Enhanced Agent")
st.caption("Understands sentiment, entities, and intent before responding")

# -------------------------------------------------------------------
# 4. NLP helper functions (unchanged)
# -------------------------------------------------------------------
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def detect_intent(text, entities):
    text_lower = text.lower()
    if any(word in text_lower for word in ["hello", "hi", "hey"]):
        return "greeting"
    elif any(word in text_lower for word in ["bye", "goodbye", "exit"]):
        return "farewell"
    elif any(word in text_lower for word in ["help", "what can you do", "capabilities"]):
        return "help"
    elif "weather" in text_lower:
        return "weather_query"
    elif any(word in text_lower for word in ["thank", "thanks"]):
        return "gratitude"
    else:
        return "general"

def clean_messages_for_api(messages):
    """Remove nlp_insights and any other non‑standard fields."""
    cleaned = []
    for msg in messages:
        cleaned.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    return cleaned

# -------------------------------------------------------------------
# 5. Session state & chat history
# -------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "nlp_insights" in msg:
            with st.expander("🔍 NLP Insights (what the agent understood)"):
                st.json(msg["nlp_insights"])

# -------------------------------------------------------------------
# 6. Chat input and response generation
# -------------------------------------------------------------------
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run NLP analysis
    sentiment = analyze_sentiment(prompt)
    entities = extract_entities(prompt)
    intent = detect_intent(prompt, entities)

    nlp_insights = {
        "sentiment": sentiment,
        "entities": [{"text": ent[0], "label": ent[1]} for ent in entities],
        "intent": intent
    }

    system_prompt = f"""
You are a helpful assistant. The user's sentiment is {sentiment}. Their intent is {intent}.
If they mentioned any entities: {entities}
Use this information to tailor your response appropriately.
"""

    # Generate response with manual streaming
    with st.chat_message("assistant"):
        try:
            # Clean messages (remove nlp_insights) before sending to Groq
            api_messages = clean_messages_for_api(st.session_state.messages)

            stream = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *api_messages
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )

            full_response = ""
            response_placeholder = st.empty()

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)

        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            st.error(full_response)

    # Save assistant message with NLP insights (for display only)
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "nlp_insights": nlp_insights
    })