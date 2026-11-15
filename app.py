"""Streamlit front end:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from report import DISCLAIMER
from run import run

st.set_page_config(page_title="NeuroMotion v1", layout="wide")
st.title("NeuroMotion v1 - Human Body Understanding System")
st.caption(DISCLAIMER)

with st.expander("How to use", expanded=not st.session_state.get("seen")):
    st.markdown(
        "1. Tick **Use synthetic demo** to see the pipeline work with no video.\n"
        "2. Or upload a 10-20 s clip: one person, full body in frame, side-on, steady camera, good light.\n"
        "3. Use the **lite** model on this hosted app; clips over ~30 s may be slow.\n"
        "4. Read the observations, plots and annotated video, then download the report.\n\n"
        "**Privacy:** this is a public demo. Upload only your own or consented clips, never patient video."
    )
    st.session_state["seen"] = True

variant = st.sidebar.selectbox("Pose model", ["lite", "full", "heavy"], index=0)
use_demo = st.sidebar.checkbox("Use synthetic demo instead of a video")
asym = st.sidebar.checkbox("Demo: reduce right knee motion", disabled=not use_demo)
up = None if use_demo else st.file_uploader("Upload a full-body video (side view works best)",
                                            type=["mp4", "mov", "avi", "mkv"])

if use_demo or up:
    out = Path(tempfile.mkdtemp())
    bar = st.progress(0.0, text="Analysing...")
    if up:
        vid = out / up.name
        vid.write_bytes(up.getbuffer())
        _, angles, res = run(video=vid, out=out, variant=variant, annotate=True,
                             progress=lambda p: bar.progress(p, text="Estimating pose..."))
    else:
        _, angles, res = run(demo=True, out=out, asymmetric_demo=asym)
    bar.empty()

    c1, c2, c3 = st.columns(3)
    c1.metric("Movement type", res["movement_type_heuristic"])
    c2.metric("Cadence (steps/min)", f"{res['gait']['cadence_steps_per_min']:.0f}")
    c3.metric("Duration (s)", f"{res['duration_s']:.1f}")

    st.subheader("Observations")
    for o in res["observations"]:
        st.write("-", o)
    st.image(str(out / "skeleton_frames.png"))
    st.image(str(out / "joint_angles.png"))
    if (out / "annotated.mp4").exists():
        st.video(str(out / "annotated.mp4"))
    st.download_button("Download report (Markdown)", (out / "report.md").read_text(), "report.md")
    st.download_button("Download report (JSON)", (out / "report.json").read_text(), "report.json")
