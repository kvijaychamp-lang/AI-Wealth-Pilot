"""Reusable Streamlit dashboard components."""
import streamlit as st


def render_market_alert(message: str, status: str = 'info') -> None:
    """Display a styled status alert in the dashboard."""
    if status == 'warning':
        st.warning(message)
    elif status == 'error':
        st.error(message)
    else:
        st.info(message)


def render_metric(name: str, value: str) -> None:
    """Render a simple metric value in the dashboard."""
    st.metric(label=name, value=value)
