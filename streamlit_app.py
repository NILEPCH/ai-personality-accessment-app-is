import streamlit as st
import openai as genai
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from google.cloud import bigquery
from google.oauth2.service_account import Credentials

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "user_selection"

def navigate_to(page):
    st.session_state.page = page

# Page: User Selection
def user_selection_page():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Welcome to the Recruitment Portal</h1>", unsafe_allow_html=True)
    st.write("Please select whether you are a **Candidate** or **HR** and press Submit to proceed.")

    with st.form("selection_form"):
        user_type = st.radio("Select your role:", ["Candidate", "HR"], index=0)
        submitted = st.form_submit_button("Submit")

        if submitted:
            if user_type == "Candidate":
                navigate_to("candidate_form")
            elif user_type == "HR":
                navigate_to("hr_form")

# Page: Candidate Form
def candidate_form_page():
    st.markdown("<h2 style='text-align: center; color: #2196F3;'>Candidate Information</h2>", unsafe_allow_html=True)
    st.write("Please provide your details below.")

    with st.form("candidate_form"):
        name = st.text_input("Name")
        surname = st.text_input("Surname")
        age = st.number_input("Age", min_value=18, max_value=99, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        applied_position = st.text_input("Applied Position")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if not all([name, surname, age, gender, applied_position]):
                st.error("Please fill out all fields!")
            else:
                st.session_state.candidate_data = {
                    "name": name,
                    "surname": surname,
                    "age": age,
                    "gender": gender,
                    "applied_position": applied_position
                }
                navigate_to("AI_retrieval_personality_assessment_page")

# Page: HR Login Form
def hr_form_page():
    st.markdown("<h2 style='text-align: center; color: #FF5722;'>HR Login</h2>", unsafe_allow_html=True)
    st.write("Please log in using your HR credentials.")

    with st.form("hr_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        gemini_api_key = st.text_input("Gemini API Key", placeholder="Type your API Key here...", type="password")
        submitted = st.form_submit_button("Submit")

        if submitted:
            if username == "HR001" and password == "MADT9000" and gemini_api_key:
                st.session_state.gemini_api_key = gemini_api_key
                navigate_to("chatbot_chat_with_candidate_result_page")
            else:
                st.error("Incorrect credentials or missing API Key!")

@st.cache_resource
def load_models():
    # Pace model and tokenizer
    pace_model_name = "nileycena/disc_pace_roberta"
    pace_model = AutoModelForSequenceClassification.from_pretrained(pace_model_name)
    pace_tokenizer = AutoTokenizer.from_pretrained(pace_model_name)

    # Focus model and tokenizer
    focus_model_name = "nileycena/disc_focus_roberta"
    focus_model = AutoModelForSequenceClassification.from_pretrained(focus_model_name)
    focus_tokenizer = AutoTokenizer.from_pretrained(focus_model_name)

    # Similarity model
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    return pace_model, pace_tokenizer, focus_model, focus_tokenizer, similarity_model

def AI_retrieval_personality_assessment_page():
    pace_model, pace_tokenizer, focus_model, focus_tokenizer, similarity_model = load_models()

    # Define 7 Questions
    pace_questions = [
        "How do you handle deadlines?",
        "When starting a new task, how do you begin?",
        "How do you approach decision-making?",
        "How do you typically respond to conflict?",
        "When presented with a new idea, what is your initial reaction?",
        "How do you handle a situation where you need to change your plan?",
        "How do you prefer to receive instructions?",
    ]

    def is_relevant(response, question):
        response_embedding = similarity_model.encode(response, convert_to_tensor=True)
        question_embedding = similarity_model.encode(question, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(response_embedding, question_embedding)
        return similarity_score.item() > 0.5

    def classify_response(response):
        # Pace classification
        pace_inputs = pace_tokenizer(response, return_tensors="pt", truncation=True, padding=True, max_length=128)
        pace_outputs = pace_model(**pace_inputs)
        pace_logits = pace_outputs.logits
        pace_probs = F.softmax(pace_logits, dim=1).detach().numpy()[0]
        pace_label = "slow" if torch.argmax(pace_logits, axis=1).item() == 0 else "fast"

        # Focus classification
        focus_inputs = focus_tokenizer(response, return_tensors="pt", truncation=True, padding=True, max_length=128)
        focus_outputs = focus_model(**focus_inputs)
        focus_logits = focus_outputs.logits
        focus_probs = F.softmax(focus_logits, dim=1).detach().numpy()[0]
        focus_label = "task" if torch.argmax(focus_logits, axis=1).item() == 0 else "people"

        # Calculate DISC distribution
        p_slow = pace_probs[0]
        p_fast = pace_probs[1]
        p_task = focus_probs[0]
        p_people = focus_probs[1]

        D = p_fast * p_task
        I = p_fast * p_people
        S = p_slow * p_people
        C = p_slow * p_task

        return {
            "pace_label": pace_label,
            "focus_label": focus_label,
            "D": D,
            "I": I,
            "S": S,
            "C": C
        }

    def chatbot():
        st.title("DISC Chatbot System")
        st.write("Answer 7 DISC-Personality Assessment Questions. Each answer must be at least 20 words.")

        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
            st.session_state.responses = []

        if st.session_state.current_question < len(pace_questions):
            question = pace_questions[st.session_state.current_question]
            st.subheader(f"Question {st.session_state.current_question + 1}:")
            st.write(question)

            response_key = f"q{st.session_state.current_question}"
            response = st.text_area("Your answer:", key=response_key)

            # Show current word count
            word_count = len(response.strip().split())
            st.write(f"Current word count: {word_count}/20 words")

            if st.button("Submit", key=f"btn{st.session_state.current_question}"):
                if not response.strip():
                    st.warning("Please provide an answer before submitting.")
                else:
                    if word_count < 20:
                        st.warning(f"Your answer is too short. Please provide at least 20 words. Currently, you have {word_count} words.")
                    elif not is_relevant(response, question):
                        st.warning("Your answer is not relvant to the question. please answer again")
                    else:
                        classification_results = classify_response(response)
                        st.session_state.responses.append({
                            "question": question,
                            "response": response,
                            "pace_label": classification_results["pace_label"],
                            "focus_label": classification_results["focus_label"],
                            "D": classification_results["D"],
                            "I": classification_results["I"],
                            "S": classification_results["S"],
                            "C": classification_results["C"]
                        })
                        st.session_state.current_question += 1
        else:
            # All questions answered
            st.success("All questions answered!")
            st.write("Thank you very much for your time.")

            # Calculate average D, I, S, C
            total_D = sum(r["D"] for r in st.session_state.responses)
            total_I = sum(r["I"] for r in st.session_state.responses)
            total_S = sum(r["S"] for r in st.session_state.responses)
            total_C = sum(r["C"] for r in st.session_state.responses)

            num_questions = len(st.session_state.responses)
            avg_D = total_D / num_questions
            avg_I = total_I / num_questions
            avg_S = total_S / num_questions
            avg_C = total_C / num_questions

            # Convert to percentage
            D_percent = avg_D * 100
            I_percent = avg_I * 100
            S_percent = avg_S * 100
            C_percent = avg_C * 100

            results = {
                "D": D_percent,
                "I": I_percent,
                "S": S_percent,
                "C": C_percent
            }
            top_dimension = max(results, key=results.get)

            st.write(f"Your DiSC result is: **{top_dimension}**")
            st.write(f"D = {D_percent:.2f}%")
            st.write(f"I = {I_percent:.2f}%")
            st.write(f"S = {S_percent:.2f}%")
            st.write(f"C = {C_percent:.2f}%")

            # แสดงผลลัพธ์เป็นตาราง: Question#, Question, Response, Pace, Focus
            df_data = []
            for idx, ans in enumerate(st.session_state.responses, start=1):
                df_data.append({
                    "Question Number": idx,
                    "Question": ans["question"],
                    "Response": ans["response"],
                    "Pace": ans["pace_label"],
                    "Focus": ans["focus_label"]
                })
            df = pd.DataFrame(df_data)
            st.table(df)

    chatbot()

# Page: Chatbot Chat with Candidate Results
def chatbot_chat_with_candidate_result_page():
    st.markdown("<h2 style='text-align: center; color: #009688;'>Chat with Candidate Results</h2>", unsafe_allow_html=True)
    st.write("Let's start a conversation with the candidate.")

    gemini_api_key = st.session_state.get("gemini_api_key")
    model = None

    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel("gemini-pro")
            st.success("Gemini API Key successfully configured.")
        except Exception as e:
            st.error(f"Error setting up the Gemini model: {e}")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, message in st.session_state.chat_history:
        st.chat_message(role).markdown(message)

    if user_input := st.chat_input("Type your message here..."):
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        if model:
            try:
                personality_prompt = "You need to summarize what you know about the candidate from the stored data only."
                full_input = f"{personality_prompt}\nUser: {user_input}\nAssistant:"
                response = model.generate_content(full_input)
                bot_response = response.text
                st.session_state.chat_history.append(("assistant", bot_response))
                st.chat_message("assistant").markdown(bot_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Provide a valid Gemini API Key to start the conversation.")

# Page Navigation
if st.session_state.page == "user_selection":
    user_selection_page()
elif st.session_state.page == "candidate_form":
    candidate_form_page()
elif st.session_state.page == "hr_form":
    hr_form_page()
elif st.session_state.page == "AI_retrieval_personality_assessment_page":
    AI_retrieval_personality_assessment_page()
elif st.session_state.page == "chatbot_chat_with_candidate_result_page":
    chatbot_chat_with_candidate_result_page()
