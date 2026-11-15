"""Streamlit front end:  streamlit run app.py"""
import tempfile
from pathlib import Path

import streamlit as st

from report import DISCLAIMER
from run import run
from webui import MAX_EMBED_VIDEO_BYTES, png_html, video_html

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


def _collect(out: Path, res: dict) -> dict:
    def read(name):
        p = out / name
        return p.read_bytes() if p.exists() else None

    return {"res": res, "skeleton": read("skeleton_frames.png"), "angles": read("joint_angles.png"),
            "md": (out / "report.md").read_text(), "json": (out / "report.json").read_text(),
            "video": read("annotated.mp4")}


@st.cache_data(show_spinner=False, max_entries=2)
def analyse_upload(data: bytes, suffix: str, variant: str) -> dict:
    out = Path(tempfile.mkdtemp())
    vid = out / f"input{suffix}"
    vid.write_bytes(data)
    _, _, res = run(video=vid, out=out, variant=variant, annotate=True)
    return _collect(out, res)


@st.cache_data(show_spinner=False, max_entries=2)
def analyse_demo(asym: bool) -> dict:
    out = Path(tempfile.mkdtemp())
    _, _, res = run(demo=True, out=out, asymmetric_demo=asym)
    return _collect(out, res)


variant = st.sidebar.selectbox("Pose model", ["lite", "full", "heavy"], index=0)
use_demo = st.sidebar.checkbox("Use synthetic demo instead of a video")
asym = st.sidebar.checkbox("Demo: reduce right knee motion", disabled=not use_demo)
up = None if use_demo else st.file_uploader("Upload a full-body video (side view works best)",
                                            type=["mp4", "mov", "avi", "mkv"])

data = None
if use_demo:
    data = analyse_demo(asym)
elif up:
    try:
        with st.spinner("Estimating pose and analysing movement... (first run downloads the pose model)"):
            data = analyse_upload(up.getvalue(), Path(up.name).suffix or ".mp4", variant)
    except Exception as e:  # show a readable message instead of a redacted traceback
        st.error(f"Could not analyse this video: {type(e).__name__}: {e}")

if data:
    res = data["res"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Movement type", res["movement_type_heuristic"])
    c2.metric("Cadence (steps/min)", f"{res['gait']['cadence_steps_per_min'] or 0:.0f}")
    c3.metric("Duration (s)", f"{res['duration_s']:.1f}")

    st.subheader("Observations")
    for o in res["observations"]:
        st.write("-", o)

    for key, alt in (("skeleton", "Skeleton snapshots"), ("angles", "Joint angles and gait events")):
        if data[key]:
            st.markdown(png_html(data[key], alt), unsafe_allow_html=True)

    if data["video"]:
        if len(data["video"]) <= MAX_EMBED_VIDEO_BYTES:
            st.markdown(video_html(data["video"]), unsafe_allow_html=True)
        else:
            st.info("Annotated video is large; use the download button below.")
        st.download_button("Download annotated video", data["video"], "annotated.mp4", "video/mp4")
    st.download_button("Download report (Markdown)", data["md"], "report.md")
    st.download_button("Download report (JSON)", data["json"], "report.json")
