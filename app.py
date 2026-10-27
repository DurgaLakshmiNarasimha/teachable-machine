"""Streamlit UI for the Teachable Machine. Run: streamlit run app.py"""
import streamlit as st

from ml import TeachableMachine

st.set_page_config(page_title="Teachable Machine", page_icon="🧠", layout="wide")
st.title("🧠 Teachable Machine")
st.caption("Teach the app to recognise your own images — upload or capture examples per class, train, then test.")

ss = st.session_state
ss.setdefault("data", {})      # {class: [bytes]}
ss.setdefault("tm", TeachableMachine())
ss.setdefault("stats", None)

left, right = st.columns([3, 2])

with left:
    st.subheader("1. Add classes & examples")
    new = st.text_input("New class name")
    if st.button("Add class") and new.strip():
        ss.data.setdefault(new.strip(), [])
    for cls in list(ss.data):
        with st.expander(f"{cls} — {len(ss.data[cls])} example(s)", expanded=True):
            ups = st.file_uploader("Upload images", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key=f"up_{cls}")
            if st.button(f"Save uploaded images to '{cls}'", key=f"save_{cls}") and ups:
                ss.data[cls] += [u.getvalue() for u in ups]
                st.rerun()
            cam = st.camera_input("…or use webcam", key=f"cam_{cls}")
            if cam is not None and st.button(f"Add webcam photo to '{cls}'", key=f"camadd_{cls}"):
                ss.data[cls].append(cam.getvalue())
                st.rerun()
            if ss.data[cls]:
                st.image(ss.data[cls][-6:], width=70)
            if st.button(f"Delete class '{cls}'", key=f"del_{cls}"):
                del ss.data[cls]; st.rerun()

    st.subheader("2. Train")
    if st.button("Train model", type="primary"):
        try:
            ss.stats = ss.tm.train(ss.data)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    if ss.stats:
        acc = ss.stats["cv_accuracy"]
        st.success(f"Trained on {ss.stats['samples']} images. " + (f"Cross-validated accuracy: {acc:.0%}" if acc is not None else ""))
        ss.tm.save("model.joblib")
        with open("model.joblib", "rb") as fh:
            st.download_button("Download trained model", fh.read(), "model.joblib")

with right:
    st.subheader("3. Test")
    if ss.tm.model is None:
        st.info("Train the model to enable predictions.")
    else:
        test_up = st.file_uploader("Test image", type=["jpg", "jpeg", "png", "webp"], key="test_up")
        test_cam = st.camera_input("Or test with webcam", key="test_cam")
        src = test_up or test_cam
        if src is not None:
            st.image(src.getvalue(), width=250)
            for label, p in ss.tm.predict(src.getvalue()).items():
                st.write(f"**{label}** — {p:.0%}")
                st.progress(min(max(p, 0.0), 1.0))
