from __future__ import annotations

from typing import List

from ... import AutoEvalReport

try:
    import streamlit as st
except Exception:  # pragma: no cover
    raise

from ...pipelines.autoeval_plotting import (
    plot_comparison,
    plot_delta_comparison,
    aggregate_dfs,
    fig_to_png_bytes,
    fig_to_pdf_bytes,
)


def render_compare(active: List[AutoEvalReport]):
    st.subheader("Compare Multiple Reports")
    if len(active) < 2:
        st.info("Load at least two reports to compare.")
        return

    labels = [f"{report.embedder_name} - ({report.training_date})" for report in active]
    options = list(range(len(active)))
    idxs = st.multiselect("Select reports", default=options, options=options,
                          format_func=lambda i: labels[i])

    if len(idxs) == 1:
        st.info("Select at least two reports to compare.")
        st.stop()

    chosen = [active[i] for i in idxs]
    dev_mode = st.session_state.state.get_development_mode()

    # Baseline selection
    baseline_options = [report.embedder_name for report in chosen]
    baseline_model = st.selectbox("Select baseline model for delta plots", options=baseline_options)

    # Supervised comparison
    st.markdown("#### Supervised (PBC)")
    df_sup = aggregate_dfs([
        report.supervised_results["PBC"].to_df(framework="PBC", development_mode=dev_mode).assign(Model=report.embedder_name)
        for report in chosen if "PBC" in report.supervised_results
    ])
    if df_sup is None or df_sup.empty:
        st.caption("No overlapping supervised tasks to compare.")
    else:
        # Wide table similar to previous pivot
        pivot = (
            df_sup.pivot_table(index=["Task", "Test Set", "Metric"], columns="Model", values="Mean", aggfunc="first",
                               sort=False)
            .reset_index()
        )
        st.dataframe(pivot, use_container_width=True)
        
        st.markdown("**Absolute Comparison**")
        fig, ax = plot_comparison(df_sup)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig), file_name="supervised_comparison.png",
                               mime="image/png", key="sup_abs_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig), file_name="supervised_comparison.pdf",
                               mime="application/pdf", key="sup_abs_pdf")
        else:
            st.info("Install 'matplotlib' and 'seaborn' to render the comparison plot.")

        st.markdown(f"**Delta Comparison (Baseline: {baseline_model})**")
        fig_delta, ax_delta = plot_delta_comparison(df_sup, baseline_model)
        if fig_delta is not None:
            st.pyplot(fig_delta, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig_delta),
                               file_name="supervised_delta_comparison.png",
                               mime="image/png", key="sup_delta_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig_delta),
                               file_name="supervised_delta_comparison.pdf",
                               mime="application/pdf", key="sup_delta_pdf")
        else:
            st.info("Select a baseline model and ensure overlapping tasks to render the delta plot.")

    # Zeroshot comparison
    st.markdown("#### Zero-Shot (PGYM)")
    df_zero = aggregate_dfs([
        report.zeroshot_results["PGYM"].to_df(framework="PGYM", development_mode=dev_mode).assign(Model=report.embedder_name)
        for report in chosen if "PGYM" in report.zeroshot_results
    ])
    if df_zero is None or df_zero.empty:
        st.caption("No overlapping zero-shot tasks to compare.")
    else:
        pivot = (
            df_zero.pivot_table(index=["Task", "Metric"], columns="Model", values="Mean", aggfunc="first", sort=False)
            .reset_index()
        )
        st.dataframe(pivot, use_container_width=True)
        
        st.markdown("**Absolute Comparison**")
        fig, ax = plot_comparison(df_zero)
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig), file_name="zeroshot_comparison.png",
                               mime="image/png", key="zero_abs_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig), file_name="zeroshot_comparison.pdf",
                               mime="application/pdf", key="zero_abs_pdf")
        else:
            st.info("Install 'matplotlib' and 'seaborn' to render the comparison plot.")

        st.markdown(f"**Delta Comparison (Baseline: {baseline_model})**")
        fig_delta, ax_delta = plot_delta_comparison(df_zero, baseline_model)
        if fig_delta is not None:
            st.pyplot(fig_delta, use_container_width=True)
            st.download_button("⬇️ Download PNG", data=fig_to_png_bytes(fig_delta),
                               file_name="zeroshot_delta_comparison.png",
                               mime="image/png", key="zero_delta_png")
            st.download_button("⬇️ Download PDF", data=fig_to_pdf_bytes(fig_delta),
                               file_name="zeroshot_delta_comparison.pdf",
                               mime="application/pdf", key="zero_delta_pdf")
        else:
            st.info("Select a baseline model and ensure overlapping tasks to render the delta plot.")
