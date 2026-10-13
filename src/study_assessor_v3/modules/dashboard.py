"""Renders Plotly charts and Streamlit visual components for assessment outcomes."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import List, Dict, Any


class DashboardRenderer:
    """Encapsulates chart creation and Streamlit UI layout rendering."""

    @staticmethod
    def render_overview(df: pd.DataFrame):
        """Displays key summary metric cards."""
        avg_s1 = df["Study 1 Score"].mean()
        avg_s2 = df["Study 2 Score"].mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Study 1 Average Score", f"{avg_s1:.1f} / 100")
        col2.metric("Study 2 Average Score", f"{avg_s2:.1f} / 100")
        
        s1_wins = (df["Winner"] == "Study 1").sum()
        s2_wins = (df["Winner"] == "Study 2").sum()
        ties = (df["Winner"] == "Tie").sum()
        col3.metric("Win Tally (S1 / S2 / Tie)", f"{s1_wins} / {s2_wins} / {ties}")

    @staticmethod
    def render_radar_chart(df: pd.DataFrame):
        """Generates a side-by-side criteria score Radar Chart."""
        fig = go.Figure()

        categories = df["Criterion"].tolist()

        fig.add_trace(go.Scatterpolar(
            r=df["Study 1 Score"].tolist(),
            theta=categories,
            fill='toself',
            name='Study 1'
        ))
        fig.add_trace(go.Scatterpolar(
            r=df["Study 2 Score"].tolist(),
            theta=categories,
            fill='toself',
            name='Study 2'
        ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            title="Criteria Coverage Comparison (Radar Chart)"
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_bar_chart(df: pd.DataFrame):
        """Generates a side-by-side score comparison bar chart."""
        df_melted = df.melt(
            id_vars=["Criterion"],
            value_vars=["Study 1 Score", "Study 2 Score"],
            var_name="Study",
            value_name="Score"
        )
        fig = px.bar(
            df_melted,
            x="Criterion",
            y="Score",
            color="Study",
            barmode="group",
            title="Side-by-Side Assessment Scores by Criterion",
            range_y=[0, 100]
        )
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def render_traceability_table(evidence_list: List[Dict[str, Any]]):
        """Renders collapsible traceability section showing retrieved evidence."""
        st.subheader("Traceability Matrix & RAG Sources")
        for ev in evidence_list:
            with st.expander(f"[{ev['study']}] Source: {ev['filename']} (Relevance Score: {ev['relevance_score']:.3f})"):
                st.write(ev['text'])