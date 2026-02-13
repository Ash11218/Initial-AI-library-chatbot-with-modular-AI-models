import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="AI Library Chatbot", page_icon="📚")

st.title("📚 AI Library Chatbot")
st.caption("Recommendations • Learning paths • Demand prediction")

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show old messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input box
user_text = st.chat_input(
    "Ask: recommend AI books, make a learning path, predict demand..."
)


def format_response(data: dict) -> str:
    intent = data.get("intent", "UNKNOWN")
    conf = data.get("confidence", 0)
    resp = data.get("response", "")

    out = f"**Intent:** {intent}  \n**Confidence:** {conf:.2f}\n\n"

    # If recommender returns a list of dicts
    if isinstance(resp, list):
        for i, item in enumerate(resp, start=1):
            if isinstance(item, dict):
                title = item.get("title") or item.get("book") or "Unknown"
                subject = item.get("subject", "")
                difficulty = item.get("difficulty", "")
                score = item.get("score", None)
                out += f"{i}. **{title}**"
                meta = []
                if subject:
                    meta.append(subject)
                if difficulty:
                    meta.append(difficulty)
                if score is not None:
                    meta.append(f"score {score:.2f}")
                if meta:
                    out += f" — _{' | '.join(meta)}_"
                out += "\n"
            else:
                out += f"- {item}\n"
        return out

    # If demand prediction returns dict
    if isinstance(resp, dict):
        for k, v in resp.items():
            out += f"- **{k}**: {v}\n"
        return out

    # Plain string
    return out + str(resp)


if user_text:
    # show user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # call API
    with st.chat_message("assistant"):
        try:
            r = requests.post(API_URL, json={"message": user_text}, timeout=15)
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
                bot_text = f"API error {r.status_code}. Check terminal logs."
            else:
                data = r.json()
                bot_text = format_response(data)
                st.markdown(bot_text)

        except requests.exceptions.ConnectionError:
            bot_text = "❌ Can't connect to API. Start FastAPI first: `uvicorn app.main:app --reload`"
            st.error(bot_text)
        except Exception as e:
            bot_text = f"❌ Unexpected error: {e}"
            st.error(bot_text)

    st.session_state.messages.append({"role": "assistant", "content": bot_text})
