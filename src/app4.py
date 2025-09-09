import streamlit as st
import time
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

from data_extractor import run_data_extractor
from fte_analysis import run_fte_analysis
from default import main
from DatamartDaywise1 import run_daywise_tool

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, firestore

# --- Firebase setup using secrets ---
if not firebase_admin._apps:
    firebase_config = st.secrets["firebase"]
    cred = credentials.Certificate({
        "type": firebase_config["type"],
        "project_id": firebase_config["project_id"],
        "private_key_id": firebase_config["private_key_id"],
        "private_key": firebase_config["private_key"].replace("\\n", "\n"),
        "client_email": firebase_config["client_email"],
        "client_id": firebase_config["client_id"],
        "auth_uri": firebase_config["auth_uri"],
        "token_uri": firebase_config["token_uri"],
        "auth_provider_x509_cert_url": firebase_config["auth_provider_x509_cert_url"],
        "client_x509_cert_url": firebase_config["client_x509_cert_url"]
    })
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --- Session constants ---
INACTIVITY_LIMIT = 180
WARNING_DURATION = 15
REFRESH_INTERVAL = 5000

# --- Session State Init ---
st.session_state.setdefault("user_email", "")
st.session_state.setdefault("current_tool", None)
st.session_state.setdefault("start_time", None)
st.session_state.setdefault("session_start", time.time())
st.session_state.setdefault("log_data", [])
st.session_state.setdefault("last_activity", time.time())
st.session_state.setdefault("warned", False)

# --- Safe runner ---
def safe_run(label, func):
    try:
        func()
    except Exception as e:
        st.error(f"Error while running '{label}': {e}")
        st.exception(e)

# --- Activity update ---
def update_activity():
    st.session_state.last_activity = time.time()
    st.session_state.warned = False

# --- Auto-refresh heartbeat ---
st_autorefresh(interval=REFRESH_INTERVAL, key="inactivity-check")

# --- Login ---
if not st.session_state.user_email:
    st.title("🔐 Login to Start")
    email = st.text_input("Enter your email to continue:")
    if st.button("Login"):
        if email.strip():
            st.session_state.user_email = email.strip()
            st.session_state.session_start = time.time()
            update_activity()
            st.success(f"Logged in as {email}")
            st.rerun()
        else:
            st.warning("Please enter a valid email.")
    st.info("You must log in to access the tools.")
    st.stop()

# --- Inactivity Timeout ---
now = time.time()
inactive_time = now - st.session_state.last_activity

if INACTIVITY_LIMIT < inactive_time <= INACTIVITY_LIMIT + WARNING_DURATION:
    if not st.session_state.warned:
        st.warning("You’ve been inactive. Session will end in 5 seconds unless you interact.")
        st.session_state.warned = True

elif inactive_time > INACTIVITY_LIMIT + WARNING_DURATION:
    st.warning("Session ended due to inactivity.")
    end_time = now
    final_duration = end_time - st.session_state.start_time
    duration_str = time.strftime("%M:%S", time.gmtime(final_duration))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.log_data.append({
        "user": st.session_state.user_email,
        "tool": st.session_state.current_tool,
        "duration": duration_str,
        "datetime": timestamp
    })

    for entry in st.session_state.log_data:
        db.collection("whatif_user_tracking").add(entry)

    db.collection("user_sessions").add({
        "user": st.session_state.user_email,
        "total_duration": time.strftime("%M:%S", time.gmtime(end_time - st.session_state.session_start)),
        "ended_at": datetime.now().isoformat(),
        "tools_used": list({e['tool'] for e in st.session_state.log_data})
    })

    st.session_state.clear()
    st.rerun()

# --- Sidebar Navigation ---
st.sidebar.title(f"Welcome, {st.session_state.user_email}")
page = st.sidebar.radio("Select a Tool", [
    "What-if analysis", "Data Extractor", "Staffing Req Analysis", "Datamart-day(What-if)"
], on_change=update_activity)

# --- Track Tool Usage ---
current_time = time.time()
if st.session_state.current_tool != page:
    if st.session_state.current_tool is not None:
        duration = current_time - st.session_state.start_time
        duration_str = time.strftime("%M:%S", time.gmtime(duration))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.log_data.append({
            "user": st.session_state.user_email,
            "tool": st.session_state.current_tool,
            "duration": duration_str,
            "datetime": timestamp
        })
    st.session_state.current_tool = page
    st.session_state.start_time = current_time
    update_activity()

# --- Tool Execution ---
if page == "What-if analysis":
    st.title(" What-If Analysis Tool")
    safe_run("What-If Analysis", main)
elif page == "Data Extractor":
    st.title(" Data Extractor")
    safe_run("Data Extractor", run_data_extractor)
elif page == "Staffing Req Analysis":
    st.title(" Staffing Requirement Analysis")
    safe_run("Staffing Req Analysis", run_fte_analysis)
elif page == "Datamart-day(What-if)":
    st.title(" Datamart - Daywise")
    safe_run("Datamart Daywise", run_daywise_tool)

# --- Manual Session End ---
if st.sidebar.button("End Session & Save Log"):
    end_time = time.time()
    final_duration = end_time - st.session_state.start_time
    duration_str = time.strftime("%M:%S", time.gmtime(final_duration))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.log_data.append({
        "user": st.session_state.user_email,
        "tool": st.session_state.current_tool,
        "duration": duration_str,
        "datetime": timestamp
    })

    for entry in st.session_state.log_data:
        db.collection("user_tracking_logs").add(entry)

    db.collection("user_sessions").add({
        "user": st.session_state.user_email,
        "total_duration": time.strftime("%M:%S", time.gmtime(end_time - st.session_state.session_start)),
        "ended_at": datetime.now().isoformat(),
        "tools_used": list({e['tool'] for e in st.session_state.log_data})
    })

    st.sidebar.success("Session saved to Firebase")
    st.session_state.clear()
    st.rerun()
