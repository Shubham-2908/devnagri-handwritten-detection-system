import streamlit as st
import cv2
import numpy as np
from Model import Model, DecoderType
from DataLoader import Batch
from SamplePreprocessor import preprocess
import os

def load_model():
    """Load the HTR model"""
    char_list_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'charList.txt')
    with open(char_list_path, 'r', encoding='utf-8') as f:
        char_list = f.read()
    model = Model(char_list, decoderType=DecoderType.BestPath, mustRestore=True)
    return model

def main():
    st.set_page_config(page_title="Devanagari Handwriting Recognition", page_icon="📝", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .stApp {
            background-color: #000000;
        }
        .main-header {
            text-align: center;
            color: #00d4ff;
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .sub-header {
            text-align: center;
            color: #a0a0a0;
            font-size: 20px;
            margin-bottom: 30px;
        }
        .result-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(255,255,255,0.1);
            margin: 20px 0;
        }
        .result-text {
            color: white;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
            word-wrap: break-word;
        }
        .accuracy-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(255,255,255,0.1);
            margin: 20px 0;
            text-align: center;
        }
        .accuracy-label {
            color: #ffffff;
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 10px;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        }
        .accuracy-value {
            color: #ffffff;
            font-size: 56px;
            font-weight: bold;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        }
        .upload-section {
            background-color: #1a1a1a;
            padding: 30px;
            border-radius: 10px;
            border: 2px dashed #444;
            margin: 20px 0;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 18px;
            font-weight: bold;
            padding: 15px 40px;
            border-radius: 10px;
            border: none;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        .info-box {
            background-color: #1a1a1a;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #00d4ff;
            margin-top: 30px;
            color: #e0e0e0;
        }
        h3, h2, h1 {
            color: #00d4ff !important;
        }
        p, li, div {
            color: #d0d0d0 !important;
        }
        .stMarkdown {
            color: #d0d0d0;
        }
        [data-testid="stFileUploader"] {
            background-color: #1a1a1a;
            border-radius: 10px;
            padding: 10px;
        }
        [data-testid="stFileUploader"] label {
            color: #00d4ff !important;
        }
        .uploadedFile {
            background-color: #2a2a2a;
            color: #ffffff;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">📝 Devanagari Handwriting Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload an image and let AI recognize the handwritten text</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Convert uploaded file to image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display original image
            st.markdown("### 🖼️ Uploaded Image")
            st.image(image, caption='Your Image', use_container_width=True, channels="BGR")
            
            # Recognize button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button('🔍 Recognize Text', use_container_width=True):
                with st.spinner('🤖 AI is processing your image...'):
                    try:
                        # Convert to grayscale
                        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        
                        # Preprocess image
                        processed_img = preprocess(gray_image, Model.imgSize)

                        # Load model and predict
                        model = load_model()
                        
                        # Create batch
                        batch = Batch(None, [processed_img])
                        
                        # Infer
                        (recognized, probability) = model.inferBatch(batch, True)
                        
                        # Get the recognized text
                        recognized_text = recognized[0]
                        
                        # Store results in session state
                        st.session_state.recognized_text = recognized_text
                        st.session_state.probability = probability[0]
                        st.session_state.show_results = True
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f'❌ An error occurred: {str(e)}')
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
        else:
            st.info("👆 Please upload an image to get started")
    
    with col2:
        st.markdown("### ✨ Recognition Results")
        
        # Display results if available
        if hasattr(st.session_state, 'show_results') and st.session_state.show_results:
            
            # Display recognized text
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.markdown('<div style="color: white; font-size: 18px; text-align: center; margin-bottom: 10px;">Recognized Text:</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-text">{st.session_state.recognized_text}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Display accuracy with bigger font
            st.markdown('<div class="accuracy-box">', unsafe_allow_html=True)
            st.markdown('<div class="accuracy-label">Confidence Score</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="accuracy-value">{st.session_state.probability:.1%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Progress bar for visual representation
            st.progress(float(st.session_state.probability))
        else:
            # Placeholder when no results
            st.info("👉 Upload an image and click 'Recognize Text' to see results here")

    # Information section at the bottom
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### ℹ️ About this Application
    
    This application uses a **CNN-RNN neural network** with CTC loss to recognize handwritten **Devanagari** script.
    
    #### 📋 How to use:
    1. **Upload** an image containing handwritten Devanagari text (JPG, JPEG, or PNG format)
    2. Click the **'Recognize Text'** button to process the image
    3. View the **recognized text** and **confidence score** on the right panel
    
    #### 💡 Tips for best results:
    - Use clear, well-lit images
    - Ensure text is written horizontally
    - Avoid blurry or low-quality images
    - One line of text works best
    
    #### 🎯 Model Information:
    - **Architecture**: CNN (Convolutional Neural Network) + RNN (Recurrent Neural Network)
    - **Language**: Devanagari Script
    - **Training**: Pre-trained model with 27 epochs
    """)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()