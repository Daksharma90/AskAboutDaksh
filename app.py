import streamlit as st
import io
from groq import Groq
from gtts import gTTS 
from streamlit_mic_recorder import mic_recorder
import base64
import os

# --- Configuration ---
GROQ_API_KEY = st.secrets["GROQ_API_KEY"] 

GROQ_TEXT_MODEL = "llama-3.3-70b-versatile" # Model for text generation
GROQ_STT_MODEL = "whisper-large-v3" # Model for speech-to-text (Whisper)

# Placeholder for your actual photo URL. IMPORTANT: Replace this!
DAKSH_PHOTO_URL = "https://placehold.co/150x150/ffffff/1a237e?text=Daksh" 

# --- Functions for AI Interactions ---

@st.cache_data
def load_persona_data(filepath="my_persona.md"):
    """Loads the persona data from the my_persona.md file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Error: '{filepath}' not found. Please ensure your persona file is in the same directory.")
        return None

def transcribe_audio(audio_bytes_dict: dict, groq_client: Groq) -> str:
    """
    Transcribes audio bytes into text using Groq's Whisper API.
    Args:
        audio_bytes_dict (dict): Dictionary returned by mic_recorder containing audio data.
        groq_client (Groq): Initialized Groq client.
    Returns:
        str: Transcribed text.
    """
    if audio_bytes_dict is None or 'bytes' not in audio_bytes_dict:
        st.error("No valid audio data received from microphone.")
        return ""

    raw_audio_bytes = audio_bytes_dict['bytes'] # Extract raw bytes from the dictionary

    try:
        with st.spinner("Transcribing audio..."): 
            with io.BytesIO(raw_audio_bytes) as audio_file:
                audio_file.name = "audio.wav" # Groq API expects a file-like object with a name
                transcript = groq_client.audio.transcriptions.create(
                    file=(audio_file.name, audio_file.getvalue(), "audio/wav"),
                    model=GROQ_STT_MODEL,
                    response_format="text"
                )
                return transcript.strip()
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return ""

def generate_ai_response(user_text: str, persona_data: str, groq_client: Groq) -> str:
    """
    Generates a text response from Groq's Llama 3, acting as Daksh Sharma.
    Args:
        user_text (str): The user's question.
        persona_data (str): The full persona data from my_persona.md.
        groq_client (Groq): Initialized Groq client.
    Returns:
        str: AI-generated response.
    """
    try:
        # Construct the system prompt using the loaded persona data
        system_prompt = f"""
        You are an AI assistant designed to respond as Daksh Sharma would. Your goal is to accurately reflect Daksh's personality, experiences, and thought processes when answering questions.
        You have access to detailed information about Daksh Sharma below. Use this information to formulate your responses.
        If a direct answer to the user's question is not explicitly found in the provided information, infer a logical and realistic response that aligns with Daksh Sharma's overall persona, skills, interests, and professional philosophy.
        Avoid fabricating specific projects, dates, or detailed anecdotes not present in the provided persona data.
        Maintain an authentic, personal, and concise tone, typically 2-4 sentences, unless more detail is specifically requested.

        --- Daksh Sharma's Persona Data ---
        {persona_data}
        ----------------------------------
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ]

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model=GROQ_TEXT_MODEL,
            temperature=0.7, # Adjust for creativity vs. consistency
            max_tokens=250 # Limit response length for conciseness
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return "I apologize, but I encountered an issue while generating my response."

def text_to_speech(text: str) -> bytes: 
    """
    Converts text into speech audio bytes using gTTS.
    Args:
        text (str): The text to convert.
    Returns:
        bytes: Audio data in MP3 format.
    """
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return b""

# --- Streamlit UI ---
st.set_page_config(
    page_title="Daksh's Voice Persona",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for a beautiful UI - REVISED FOR DARK MODE
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #1a1a1a; /* Dark gray/black background */
        color: #f0f2f6; /* Light gray for text */
    }
    .st-emotion-cache-1cypj85 { /* Main column padding */
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .st-emotion-cache-1r6y4y9 { /* Header styling */
        color: #e0e0e0; /* Lighter gray for header */
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    .st-emotion-cache-16txt4s { /* Markdown text - Tagline */
        text-align: center;
        color: #cccccc; /* Slightly darker white for tagline */
    }
    /* Specific styling for the mic recorder button */
    div.st-emotion-cache-1j43z82 > div > button,
    div.st-emotion-cache-z5in9u > button, /* Target mic recorder button */
    .stButton > button { /* General button style */
        background-color: #4CAF50; /* Green primary color for buttons */
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Darker shadow for dark mode */
        transition: all 0.3s ease;
        border: none;
        width: 100%; /* Make buttons full width for consistency */
        margin-top: 10px; /* Add some space above buttons */
    }
    div.st-emotion-cache-1j43z82 > div > button:hover,
    div.st-emotion-cache-z5in9u > button:hover,
    .stButton > button:hover {
        background-color: #45a049; /* Darker green on hover */
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3); /* Darker shadow on hover */
    }
    .stAudio {
        border-radius: 12px;
        background-color: #333333; /* Darker background for audio player */
        padding: 10px;
        margin-top: 10px; /* Space after response text */
    }
    .stImage {
        border-radius: 50%; /* Circular image */
        border: 4px solid #4CAF50; /* Green border around image (contrasting with dark) */
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); /* Darker shadow */
    }
    .stSpinner > div > div { /* Spinner color */
        color: #4CAF50; /* Green spinner */
    }
    div.st-emotion-cache-1r4qj8m { /* text input outer div */
        border-radius: 12px;
        border: 1px solid #555555; /* Lighter border for dark mode */
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 10px;
        background-color: #333333; /* Dark background for input field */
    }
    div.st-emotion-cache-nahz7x { /* Text Area */
        border-radius: 12px;
        border: 1px solid #555555; /* Lighter border for dark mode */
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        background-color: #333333; /* Dark background for input field */
    }
    /* Style for the text output from the bot */
    p b {
        color: #4CAF50; /* Green for "You said" and "Daksh says" labels */
    }
    /* Ensure text within input fields is white */
    .st-emotion-cache-vdv0q { /* Target the actual input element for text color */
        color: #f0f2f6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Groq client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Groq client. Please check your GROQ_API_KEY in Streamlit Secrets. Error: {e}")
    st.stop() # Stop the app if API key is not set

# Load persona data
persona_data = load_persona_data()
if persona_data is None:
    st.stop() # Stop if persona data couldn't be loaded

st.markdown("<h1 class='st-emotion-cache-1r6y4y9'>👋 Meet Daksh's AI Persona!</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Changed to use a placeholder image URL that fits the color scheme
    st.image(DAKSH_PHOTO_URL, width=150, use_column_width=False) 
st.markdown("<p class='st-emotion-cache-16txt4s'>Ask me anything about Daksh's life, career, or aspirations. I'll respond as he would!</p>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("<h3>🎙️ Speak Your Question</h3>", unsafe_allow_html=True)

# Microphone recorder
audio_data_dict = mic_recorder( 
    start_prompt="Click to Speak",
    stop_prompt="Stop Speaking",
    key='mic_recorder',
    format="wav"
)

user_text = ""
ai_response = ""
audio_output = None # Initialize audio_output to None

# Check if audio_data_dict is not None and contains 'bytes'
if audio_data_dict: 
    user_text = transcribe_audio(audio_data_dict, groq_client) 
    if user_text:
        st.markdown(f"<p><b>You said:</b> {user_text}</p>", unsafe_allow_html=True)
        st.session_state['last_user_text'] = user_text 

        ai_response = generate_ai_response(user_text, persona_data, groq_client)
        if ai_response:
            st.markdown(f"<p><b>Daksh says:</b> {ai_response}</p>", unsafe_allow_html=True)
            audio_output = text_to_speech(ai_response) # Assign audio_output here
            
            # Now, check if audio_output is not None before playing
            if audio_output:
                st.audio(audio_output, format='audio/mp3', start_time=0, autoplay=True) 
    else:
        st.warning("Could not transcribe your audio. Please try again.")

st.markdown("---")
st.markdown("<h3>⌨️ Or Type Your Question</h3>", unsafe_allow_html=True)

manual_input = st.text_input("Type your question here:", key="manual_text_input")

if st.button("Get Response (Text Only)"):
    if manual_input:
        ai_response = generate_ai_response(manual_input, persona_data, groq_client)
        if ai_response:
            st.markdown(f"<p><b>Daksh says:</b> {ai_response}</p>", unsafe_allow_html=True)
    else:
        st.warning("Please type a question to get a text response.")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.8em; color: #777;'>Powered by Groq Llama 3, Groq Whisper, and gTTS.</p>", unsafe_allow_html=True)

