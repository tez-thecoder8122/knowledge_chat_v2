import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change if deployed elsewhere
st.set_page_config(page_title="Knowledge Chat System", layout="wide")

# Session state for auth token
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

# Helper: API Request with Auth
def api_request(method, endpoint, data=None, files=None):
    url = f"{API_BASE_URL}{endpoint}"
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    if method == "POST":
        if files:
            response = requests.post(url, headers=headers, files=files)
        else:
            headers["Content-Type"] = "application/json"
            response = requests.post(url, headers=headers, json=data)
    elif method == "GET":
        response = requests.get(url, headers=headers, params=data)
    
    return response

# Sidebar: Authentication
st.sidebar.title("🔐 Authentication")
if not st.session_state.token:
    with st.sidebar.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")
        
        if login_button:
            response = api_request("POST", "/auth/login", data={"username": username, "password": password})
            if response.status_code == 200:
                result = response.json()
                st.session_state.token = result["access_token"]
                st.session_state.username = username
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error("❌ Login failed. Check your credentials.")
else:
    st.sidebar.success(f"👤 Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.username = None
        st.rerun()

# Main App (only if authenticated)
if st.session_state.token:
    st.title("📚 Knowledge Chat System")
    st.markdown("Upload documents and ask questions to get AI-powered answers with images and tables!")
    
    # Tab layout
    tab1, tab2 = st.tabs(["📤 Upload Documents", "💬 Ask Questions"])
    
    # Tab 1: Upload
    with tab1:
        st.header("Upload PDF Documents")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        
        if uploaded_file and st.button("Upload Document"):
            with st.spinner("Uploading and processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = api_request("POST", "/documents/upload", files=files)
                
                if response.status_code == (200, 201):
                    result = response.json()
                    st.success(f"✅ Document uploaded: **{result['original_filename']}**")
                    # Convert the dict to a DataFrame and display as a table (exclude 'message' if desired)
                    table_data = {k: v for k, v in result.items() if k != "message"}
                    df = pd.DataFrame([table_data])
                    st.table(df)
                else:
                    st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
    
    # Tab 2: Query
    with tab2:
        st.header("Ask Questions About Your Documents")
        
        question = st.text_input("Enter your question:")
        top_k = st.slider("Number of context chunks to retrieve", 1, 10, 3)
        
        if st.button("Get Answer") and question:
            with st.spinner("Searching and generating answer..."):
                payload = {
                    "question": question,
                    "top_k": top_k,
                    "include_media": True
                }
                response = api_request("POST", "/api/query/ask", data=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display Answer
                    st.subheader("💡 Answer")
                    st.write(result["answer"])
                    
                    # Display Text Sources
                    st.subheader("📄 Text Sources")
                    for idx, source in enumerate(result["text_sources"], 1):
                        with st.expander(f"Source {idx}: {source['document']} (Distance: {source['distance']:.4f})"):
                            st.write(source["chunk"])
                    
                    # Display Media Items (Images & Tables)
                    if result.get("media_items"):
                        st.subheader("🖼️ Related Media")
                        
                        for media in result["media_items"]:
                            if media["type"] == "image":
                                st.markdown(f"**Image from Page {media['page_number']}**")
                                if media.get("description"):
                                    st.caption(media["description"])
                                
                                # Decode and display base64 image
                                if media.get("image_base64"):
                                    img_data = base64.b64decode(media["image_base64"])
                                    img = Image.open(BytesIO(img_data))
                                    st.image(img, use_container_width=True)
                            
                            elif media["type"] == "table":
                                st.markdown(f"**Table from Page {media['page_number']}**")
                                if media.get("description"):
                                    st.caption(media["description"])
                                
                                # Display table HTML
                                if media.get("table_html"):
                                    st.markdown(media["table_html"], unsafe_allow_html=True)
                                
                                # Option to download CSV
                                if media.get("table_csv"):
                                    st.download_button(
                                        label="Download Table as CSV",
                                        data=media["table_csv"],
                                        file_name=f"table_page_{media['page_number']}.csv",
                                        mime="text/csv"
                                    )
                    else:
                        st.info("ℹ️ No related media found for this query.")
                    
                    # Context used
                    with st.expander("📋 Full Context Used"):
                        for idx, ctx in enumerate(result["context_used"], 1):
                            st.write(f"**Chunk {idx}:** {ctx}")
                
                else:
                    st.error(f"❌ Query failed: {response.json().get('detail', 'Unknown error')}")

else:
    st.info("👈 Please login using the sidebar to access the application.")
