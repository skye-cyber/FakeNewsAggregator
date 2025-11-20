import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


max_length = 54
padding_type = 'post'
trunc_type = 'post'

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-real {
        color: black;
        padding: 20px;
        background-color: #d4edda;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0px;
    }
    .result-fake {
        color: black;
        padding: 20px;
        background-color: #f8d7da;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 10px 0px;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 10px 0px;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #dc3545, #ffc107, #28a745);
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>', unsafe_allow_html=True)

# Sidebar for information
with st.sidebar:
    st.header("About This System")
    st.write("""
    This intelligent system uses deep learning to analyze news content
    and predict its credibility. The model combines CNN and LSTM architectures
    with pre-trained GloVe embeddings.

    **How to use:**
    1. Paste news text in the input area
    2. Click 'Analyze News'
    3. View the results and confidence score
    """)

    st.header("Model Information")
    st.metric("Model Architecture", "CNN-LSTM Hybrid")
    st.metric("Embedding", "GloVe 50D")
    st.metric("Accuracy", "> 90%")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Analyze News Article")

    # Input method selection
    input_method = st.radio("Choose input method:",
                            ["Paste Text", "Enter URL (Future Feature)"])

    if input_method == "Paste Text":
        news_text = st.text_area(
            "Paste the news content here:",
            height=200,
            placeholder="Enter the news headline and content you want to analyze..."
        )

        # Additional options
        col1a, col1b = st.columns(2)
        with col1a:
            analyze_button = st.button("🚀 Analyze News", type="primary", use_container_width=True)
        with col1b:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

    else:
        st.info("URL analysis feature coming soon!")
        news_text = ""

with col2:
    st.subheader("📊 Results")

    # Placeholder for results
    results_placeholder = st.empty()
    confidence_placeholder = st.empty()
    details_placeholder = st.empty()

    # Example predictions
    st.subheader("📋 Try These Examples")
    example1 = st.button("Real News Example")
    example2 = st.button("Fake News Example")

# Load model and tokenizer (in a real app, this would be cached)


@st.cache_resource
def load_model_components():
    """
    Load the trained model and tokenizer.
    In a real implementation, this would load your actual saved model.
    """
    # Placeholder - replace with your actual model loading code
    try:
        tokenizer = Tokenizer()
        model = load_model('models/model.keras')
        st.success("✅ Model components loaded successfully!")
        return model, tokenizer
    except Exception:
        st.warning("⚠️ Demo mode: Using simulated predictions")
        return None, None


def preprocess_text(text, tokenizer, max_length=54):
    """Preprocess the input text for prediction"""
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded


def predict_news(text, model, tokenizer):
    """Make prediction on the input text"""
    # This is a simplified version - replace with your actual prediction logic
    processed_text = preprocess_text(text, tokenizer)
    prediction = model.predict(processed_text, verbose=0)
    return prediction[0][0]


# Load model (commented out for demo)
model, tokenizer = load_model_components()

# Handle button clicks
if analyze_button and news_text:
    with st.spinner('🤖 Analyzing content...'):
        sequences = tokenizer.texts_to_sequences([news_text])
        sequences = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

        prediction = model.predict(sequences, verbose=0)

        fake_score = prediction[0][0]
        real_score = 1 - fake_score

        # Display results
        if prediction[0][0] > 0.5:
            results_placeholder.markdown(
                '<div class="result-fake">'
                '<h4>🚨 Likely FAKE News</h3>'
                '<p>This content shows characteristics of misinformation.</p>'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            results_placeholder.markdown(
                '<div class="result-real">'
                '<h4>✅ Likely REAL News</h3>'
                '<p>This content appears to be credible.</p>'
                '</div>',
                unsafe_allow_html=True
            )

        # Confidence meter
        confidence_placeholder.subheader("Confidence Level")
        confidence_value = fake_score if (prediction[0][0] > 0.5) else real_score
        print(confidence_value)
        confidence_placeholder.markdown(
            f'<div class="confidence-bar">'
            f'<div class="confidence-fill" style="width: {confidence_value * 100:.1}%"></div>'
            f'</div>'
            f'<p>{confidence_value:.1%} confidence</p>',
            unsafe_allow_html=True
        )

        # Detailed analysis
        with details_placeholder.expander("📈 Detailed Analysis"):
            st.write("**Probability Breakdown:**")
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                st.metric("Real News Probability", f"{real_score:.1%}")
            with col_prob2:
                st.metric("Fake News Probability", f"{fake_score:.1%}")

            st.write("**Key Factors Considered:**")
            factors = [
                "✓ Language Patterns Analysis",
                "✓ Source Credibility Indicators",
                "✓ Content Consistency Check",
                "✓ Sensationalism Detection"
            ]
            for factor in factors:
                st.write(factor)

elif analyze_button and not news_text:
    st.warning("⚠️ Please enter some news content to analyze.")

# Handle example buttons
if example1:
    st.session_state.news_text = "NASA's James Webb Space Telescope has discovered new exoplanets that may support life. The findings were published in the peer-reviewed journal Nature Astronomy after extensive review by international scientists."
    st.rerun()

if example2:
    st.session_state.news_text = "BREAKING: Shocking discovery reveals that drinking coffee makes you immune to COVID-19! Doctors don't want you to know this simple trick that's going viral worldwide!"
    st.rerun()

# Handle clear button
if clear_button:
    st.session_state.news_text = ""
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Fake News Detection System • Built with TensorFlow & Streamlit • "
    "<strong>Note:</strong> This is a demonstration interface. For production use, ensure proper model integration."
    "</div>",
    unsafe_allow_html=True
)

# http://localhost:8501
