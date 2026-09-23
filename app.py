"""Trading Signals Dashboard - main Streamlit entry point."""
import streamlit as st

st.set_page_config(
    page_title="Trading Signals Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui.dashboard import render_dashboard

if __name__ == "__main__":
    # Market data refreshes on page interaction or through the dashboard control.
    render_dashboard()