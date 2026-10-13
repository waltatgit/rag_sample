"""Main Streamlit execution entry point for Comparative Study Assessment Application."""

import os
import json
import pandas as pd
import streamlit as st

from llama_index.llms.openai import OpenAI
from modules.document_loader import DocumentLoader
from modules.index_manager import IndexManager
from modules.retriever import StudyRetriever
from modules.assessor import StudyAssessor
from modules.dashboard import DashboardRenderer
from modules.utils import parse_criteria_file, setup_logger

logger = setup_logger()

# Streamlit Page Setup
st.set_page_config(
    page_title="Comparative Study Assessment RAG",
    page_icon="📊",
    layout="wide"
)


def init_environment():
    """Ensure standard project storage directories exist."""
    os.makedirs("storage/study1", exist_ok=True)
    os.makedirs("storage/study2", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def get_openai_api_key() -> str:
    """Retrieves OpenAI API Key from Streamlit secrets or environment variable."""
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    elif "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        return st.secrets["openai"]["api_key"]
    elif os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    else:
        st.error(
            "OpenAI API Key not found! Please add `OPENAI_API_KEY` to `.streamlit/secrets.toml` "
            "or set it as an environment variable."
        )
        st.stop()


def main():
    init_environment()
    st.title("📊 Comparative Study Assessment RAG Application")

    # --- Retrieve API Key from Secrets ---
    api_key = get_openai_api_key()
    os.environ["OPENAI_API_KEY"] = api_key

    # --- Sidebar Configuration ---
    st.sidebar.header("Configuration & Input Setup")

    study1_folder = st.sidebar.text_input("Study 1 Folder Path", value="data/study1")
    study2_folder = st.sidebar.text_input("Study 2 Folder Path", value="data/study2")
    criteria_file = st.sidebar.text_input("Criteria Text File Path", value="data/criteria.txt")
    
    model_choice = st.sidebar.selectbox("LLM Model Choice", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    run_btn = st.sidebar.button("Run Assessment", type="primary")

    llm = OpenAI(model=model_choice, temperature=0.0)

    # --- Execution Pipeline ---
    if run_btn:
        with st.spinner("Processing documents, building/loading vector indexes, and running evaluations..."):
            try:
                # 1. Parse Criteria
                criteria = parse_criteria_file(criteria_file)
                st.sidebar.success(f"Loaded {len(criteria)} criteria.")

                # 2. Load Documents
                s1_docs = DocumentLoader(study1_folder, "Study 1").load_documents()
                s2_docs = DocumentLoader(study2_folder, "Study 2").load_documents()

                # 3. Build / Load Vector Indexes
                idx_mgr_s1 = IndexManager(storage_dir="storage/study1")
                idx_mgr_s2 = IndexManager(storage_dir="storage/study2")

                index1 = idx_mgr_s1.get_or_create_index(s1_docs)
                index2 = idx_mgr_s2.get_or_create_index(s2_docs)

                # 4. Instantiate Retrievers & Assessor
                retriever1 = StudyRetriever(index1, "Study 1", top_k=5)
                retriever2 = StudyRetriever(index2, "Study 2", top_k=5)
                assessor = StudyAssessor(llm=llm)

                # 5. Assessment loop
                detailed_results = []
                summary_rows = []
                all_traceability = []

                for idx, criterion in enumerate(criteria):
                    st.write(f"Evaluating Criterion {idx + 1}/{len(criteria)}: **{criterion}**")
                    
                    # Retrieve
                    ev1 = retriever1.retrieve_evidence(criterion)
                    ev2 = retriever2.retrieve_evidence(criterion)

                    # Collect traceability
                    for e in ev1:
                        all_traceability.append(e.dict())
                    for e in ev2:
                        all_traceability.append(e.dict())

                    # Assess
                    eval1 = assessor.evaluate_criterion(criterion, ev1)
                    eval2 = assessor.evaluate_criterion(criterion, ev2)

                    # Compare
                    comp = assessor.compare_assessments(criterion, eval1, eval2)

                    row_summary = {
                        "Criterion": criterion,
                        "Study 1 Score": eval1.score,
                        "Study 2 Score": eval2.score,
                        "Winner": comp.winner,
                        "Confidence": comp.confidence_score,
                    }
                    summary_rows.append(row_summary)

                    detailed_results.append({
                        "criterion": criterion,
                        "study1_assessment": eval1.dict(),
                        "study2_assessment": eval2.dict(),
                        "comparison": comp.dict()
                    })

                # 6. Save Artifacts to Local Persistence
                df_results = pd.DataFrame(summary_rows)
                df_results.to_csv("results/assessment_results.csv", index=False)
                
                with open("results/assessment_results.json", "w", encoding="utf-8") as f:
                    json.dump(detailed_results, f, indent=2)

                st.session_state["results_df"] = df_results
                st.session_state["results_json"] = detailed_results
                st.session_state["traceability"] = all_traceability
                st.success("Assessment Complete! Results saved to `results/` folder.")

            except Exception as e:
                logger.error(f"Execution Error: {e}", exc_info=True)
                st.error(f"Error during assessment execution: {e}")

    # --- Render Dashboard Results ---
    if "results_df" in st.session_state:
        df = st.session_state["results_df"]
        detailed = st.session_state["results_json"]
        traceability = st.session_state["traceability"]

        # Dashboard View Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            " Overall Summary", 
            " Visual Analytics", 
            " Detailed Criterion View", 
            " Source Traceability"
        ])

        with tab1:
            DashboardRenderer.render_overview(df)
            st.subheader("Summary Performance Table")
            st.dataframe(df, use_container_width=True)

        with tab2:
            col_left, col_right = st.columns(2)
            with col_left:
                DashboardRenderer.render_bar_chart(df)
            with col_right:
                DashboardRenderer.render_radar_chart(df)

        with tab3:
            st.subheader("Criterion Deep Dive")
            selected_criterion = st.selectbox(
                "Select Criterion to Inspect", 
                options=[item["criterion"] for item in detailed]
            )
            
            selected_data = next(item for item in detailed if item["criterion"] == selected_criterion)
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown("### Study 1")
                st.metric("Score", selected_data["study1_assessment"]["score"])
                st.markdown("**Strengths:**")
                for s in selected_data["study1_assessment"]["strengths"]:
                    st.markdown(f"- {s}")
                st.markdown("**Weaknesses:**")
                for w in selected_data["study1_assessment"]["weaknesses"]:
                    st.markdown(f"- {w}")
                st.markdown(f"**Rationale:** {selected_data['study1_assessment']['rationale']}")

            with col_s2:
                st.markdown("### Study 2")
                st.metric("Score", selected_data["study2_assessment"]["score"])
                st.markdown("**Strengths:**")
                for s in selected_data["study2_assessment"]["strengths"]:
                    st.markdown(f"- {s}")
                st.markdown("**Weaknesses:**")
                for w in selected_data["study2_assessment"]["weaknesses"]:
                    st.markdown(f"- {w}")
                st.markdown(f"**Rationale:** {selected_data['study2_assessment']['rationale']}")

            st.divider()
            st.markdown(f"### Comparison Winner: **{selected_data['comparison']['winner']}** (Confidence: {selected_data['comparison']['confidence_score']}%)")
            st.write(selected_data['comparison']['explanation'])

        with tab4:
            DashboardRenderer.render_traceability_table(traceability)


if __name__ == "__main__":
    main()