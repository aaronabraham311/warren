"""Warren dashboard — multi-page Streamlit entrypoint.

Run with: `streamlit run dashboard/app.py`

`set_page_config` lives here (Streamlit allows it once per run); pages must not call
it. Registering a new page is a single `st.Page(...)` line below — the History and
Eval pages (later tickets) slot in here once they exist.
"""

import sys
from pathlib import Path

# Allow `streamlit run dashboard/app.py` from the repo root to resolve `dashboard.*`
# and `storage.*` (Streamlit puts the entrypoint's own dir on sys.path, not the root).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Warren", layout="wide")

pages = [
    st.Page("pages/today.py", title="Today", icon="📊", default=True),
    # st.Page("pages/history.py", title="History", icon="🕑"),  # W5 blocks; added later
    # st.Page("pages/eval.py", title="Eval", icon="✅"),
]
st.navigation(pages).run()
