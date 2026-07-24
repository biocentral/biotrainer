from __future__ import annotations

import streamlit as st

from ..state import AutoevalSessionState


def render_evaluate_view(state: AutoevalSessionState):
    st.subheader("Evaluate")
    
    st.write("Learn how to run autoeval.")

    st.markdown("- [Autoeval Docs](https://github.com/sacdallago/biotrainer/blob/main/docs/autoeval.md) - Autoeval Documentation.")
    st.markdown("- [Autoeval Example Notebooks](https://github.com/sacdallago/biotrainer/tree/main/examples/autoeval) - Get started with autoeval example notebooks.")
