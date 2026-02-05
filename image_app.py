import numpy as np
import pandas as pd
import pickle
import streamlit as st
import cv2
import time
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fabric Quality Inspector", page_icon="🧵", layout="centered")

# --- CUSTOM CSS FOR INDUSTRIAL DESIGN ---
st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background-color: white;
    }
            
    .stSidebar {
        width: 360px !important;
    }

    .st-emotion-cache-e2kd2x h1{
        padding: 0.25rem 0px 0.25rem !important;   
    }
            
    .gradient-text {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(to right, #2BC0E4 0%, #8E2DE2 20%, #00FF87 40%, #F2994A 60%, #BDC3C7 80%, #E52D27 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
            
    .gradient-side {
        font-size: 1.5rem !important;
        font-weight: 800;
        background: linear-gradient(to right, #2BC0E4 0%, #8E2DE2 20%, #00FF87 40%, #F2994A 60%, #BDC3C7 80%, #E52D27 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
            
    .logo-side{
        font-size: 1.5rem !important;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
    }
           
    .logo{
        font-size: 3rem !important;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 10px;
        }
    
    /* Main Card Container */
    div.stButton > button:first-child {
        background-color: #004a99;
        color: white;
        border-radius: 5px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    
    /* Title Styling */
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #1c1c1c;
        font-weight: 700;
        text-align: center;
        padding-top: 20px;
    }
    
    /* Metric Card */
    .status-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #004a99;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOADING ASSETS ---
with open("image_vect.pkl", 'rb') as f:
    sc = pickle.load(f)
with open("image_dec.pkl", "rb") as f:
    knn = pickle.load(f)



def convert_img(img):
    data = cv2.imread(img)
    dt = cv2.resize(data, dsize = (300, 300), interpolation = cv2.INTER_LINEAR)
    img_arr = np.array(dt)
    img_data = img_arr.flatten()
    return img_data

# --- MAIN INTERFACE ---
st.markdown('<p class = "logo">🧵<span class = "gradient-text"> Fabric Defect AI Inspector</span></p>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Automated Visual Inspection (AVI) for Textile Quality Assurance</p>", unsafe_allow_html=True)

st.divider()


st.subheader("📤 Data Input")
uploaded_file = st.file_uploader("Upload Fabric Surface Image (JPG/PNG)", type=['jpg', 'png', 'jpeg'])

# --- SIDEBAR DOCUMENTATION ---
with st.sidebar:
    st.markdown('<p class = "logo-side">🧵<span class = "gradient-side"> Fabric Defect AI Inspector</span></p>', unsafe_allow_html=True)
    st.markdown("---")
    st.info("**Instructions:** \n1. Upload a high-res image of the fabric. \n2. Ensure lighting is uniform. \n3. Click 'Analyze Fabric'.")
    if uploaded_file:
        st.divider()
        st.success("Image Successfully Loaded..")
        st.subheader("Image Preview")
        # Showing the image in the sidebar as requested
        img_display = Image.open(uploaded_file)
        st.image(img_display, use_container_width=True, caption="Current Specimen")
    st.divider()
    st.caption("AI Engine: Decision Tree Classifier v1.2")


if uploaded_file:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

def generate_spectral_map(img_bgr):
    """Produces the high-detail multi-color feature map."""
    # 1. Convert to Gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Enhance Texture (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # 3. Apply a complex Colormap (COLORMAP_JET or COLORMAP_HSV creates that look)
    # COLORMAP_JET is the most common for the red-blue-green spectral look
    spectral_map = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
    return spectral_map


st.subheader("🔍 Analysis Output")

if uploaded_file:
    if st.button("ANALYZE FABRIC"):
        with st.spinner('Analyzing surface patterns...'):
            
            # Processing
            img_data = convert_img("temp_image.jpg")
            df = pd.DataFrame([img_data])
            transformed_data = sc.transform(df)
            
            # Mock scanning delay for UX
            time.sleep(1.5)
            prediction = knn.predict(transformed_data)[0]


            # Reading
            dt = cv2.imread("temp_image.jpg")
            defect = generate_spectral_map(dt)
            st.image(defect, use_container_width=True, caption="Defect Specimen")
            
            st.markdown("### Result Overview")
            if prediction.lower() == "defect free":
                st.success("#### ✅ QUALITY PASSED")
                st.write("No surface irregularities detected. This batch is ready for production.")
                st.metric(label="Surface Integrity", value="100%", delta="Optimal")
            else:
                st.error("#### 🚨 DEFECT DETECTED")
                st.markdown(f"**Anomaly Type:** `{prediction.upper()}`")
                st.warning("Action Required: Flag this roll for manual inspection.")
                st.metric(label="Surface Integrity", value="Critical", delta="-Fail", delta_color="inverse")
else:
    st.write("Please upload an image to begin the analysis.")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2025 Textile Systems Inc. | Industrial AI Division")