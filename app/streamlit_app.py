import time
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Auto Data Scientist", layout="centered")
st.title("🤖 Auto Data Scientist")
st.write("Upload a CSV, describe what you want predicted, and let the agents handle the rest.")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
goal = st.text_area(
    "What do you want to predict?",
    placeholder="e.g. Predict whether a tumor is malignant or benign, using the 'target' column.",
)

start_clicked = st.button("Run analysis", type="primary", disabled=not (uploaded_file and goal))

if "run_id" not in st.session_state:
    st.session_state.run_id = None

if start_clicked:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
    data = {"goal": goal}
    response = requests.post(f"{API_URL}/runs", files=files, data=data)

    if response.status_code == 200:
        st.session_state.run_id = response.json()["run_id"]
        st.success(f"Started! Run ID: {st.session_state.run_id}")
    else:
        st.error(f"Failed to start run: {response.text}")

if st.session_state.run_id:
    run_id = st.session_state.run_id
    status_placeholder = st.empty()

    with st.spinner("Agents are working on it... this can take a few minutes."):
        while True:
            status_resp = requests.get(f"{API_URL}/runs/{run_id}/status")
            status = status_resp.json()

            if status["status"] == "running":
                status_placeholder.info("Status: running...")
                time.sleep(5)
                continue

            elif status["status"] == "success":
                status_placeholder.success("Done! Pipeline completed successfully.")

                md_resp = requests.get(f"{API_URL}/runs/{run_id}/report")
                pdf_resp = requests.get(f"{API_URL}/runs/{run_id}/report.pdf")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download report (Markdown)",
                        data=md_resp.content,
                        file_name="report.md",
                        mime="text/markdown",
                    )
                with col2:
                    st.download_button(
                        "Download report (PDF)",
                        data=pdf_resp.content,
                        file_name="report.pdf",
                        mime="application/pdf",
                    )

                st.markdown("---")
                st.markdown(md_resp.content.decode("utf-8"))
                break

            else:
                status_placeholder.error(f"Run failed: {status.get('error')}")
                break
