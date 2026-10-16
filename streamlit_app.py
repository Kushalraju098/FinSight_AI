
import os
import streamlit as st
import requests
import time


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000"
).rstrip("/")


st.title("FinSight AI")

st.write(
    "Ask questions about Apple,Amazon,Google,Nvidia,Microsoft financial reports, stock prices, "
    "historical market data, and the latest financial news."
)


question = st.text_input("Enter your question here:")


def generate_response(question):
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"question": question},
            timeout=180,
        )

        response.raise_for_status()
        return response.json()["answer"]

    except requests.HTTPError as e:
        return f"API error: {e.response.status_code}\n\n{e.response.text}"

    except requests.ConnectionError:
        return "Could not connect to the API. Make sure FastAPI is running."

    except requests.Timeout:
        return "The request timed out. Try again."

    except (ValueError, KeyError):
        return "The API returned an unexpected response."


if question.strip():
    if st.button("Ask FinSight AI"):
        start_time = time.perf_counter()

        with st.spinner("Researching your question..."):
            answer = generate_response(question.strip())

        end_time = time.perf_counter()
        execution_time = end_time - start_time

        st.markdown("### Answer")
        st.write(answer)

        st.divider()
        st.metric(
                label="Execution Time",
                value=f"{execution_time:.2f} seconds",
            )
