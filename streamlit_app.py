import streamlit as st
import google.generativeai as genai
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from google.cloud import bigquery
from google.oauth2.service_account import Credentials

# NEW IMPORT FOR HUGGING FACE LOGIN
from huggingface_hub import login

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "user_selection"

def navigate_to(page):
    st.session_state.page = page


# -----------------------------------------------------------------------------
# Page: User Selection
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Page: Candidate Form
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Page: HR Login Form
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Load Models with Caching
# -----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """
    Load and return all required models for the DISC Personality test.
    This function is cached to avoid re-initializing models on every page reload.
    """

    # 1. Authenticate with Hugging Face using Streamlit Secrets
    hugging_face_token = st.secrets["hugging_face"]["token"]  # Retrieve from secrets
    login(token=hugging_face_token)  # Authenticate

    # 2. PACE model and tokenizer (for fast/slow)
    pace_model_name = "nileycena/disc_pace_roberta"
    pace_model = AutoModelForSequenceClassification.from_pretrained(pace_model_name)
    pace_tokenizer = AutoTokenizer.from_pretrained(pace_model_name)

    # 3. FOCUS model and tokenizer (for people/task)
    focus_model_name = "nileycena/disc_focus_roberta"
    focus_model = AutoModelForSequenceClassification.from_pretrained(focus_model_name)
    focus_tokenizer = AutoTokenizer.from_pretrained(focus_model_name)

    # 4. Similarity model (Sentence Transformer)
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    return pace_model, pace_tokenizer, focus_model, focus_tokenizer, similarity_model


# -----------------------------------------------------------------------------
# Page: AI Retrieval + Personality Assessment
# -----------------------------------------------------------------------------
def AI_retrieval_personality_assessment_page():
    # Load all necessary models
    pace_model, pace_tokenizer, focus_model, focus_tokenizer, similarity_model = load_models()

    # 1. Split questions into two categories
    PACE_QUESTIONS = [
        "How do you handle deadlines?",
        "When starting a new task, how do you begin?",
        "How do you handle a situation where you need to change your plan?"
    ]

    FOCUS_QUESTIONS = [
        "How do you approach decision-making?",
        "How do you typically respond to conflict?",
        "When presented with a new idea, what is your initial reaction?"
    ]

    # 2. Combine the questions into a single list to iterate in sequence
    all_questions = PACE_QUESTIONS + FOCUS_QUESTIONS

    # Helper function to check relevance via similarity model
    def is_relevant(response, question):
        response_embedding = similarity_model.encode(response, convert_to_tensor=True)
        question_embedding = similarity_model.encode(question, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(response_embedding, question_embedding)
        return similarity_score.item() > 0.5

    # Classification function, branches on whether question is PACE or FOCUS
    def classify_response(question, response):
        if question in PACE_QUESTIONS:
            # Use pace model for fast/slow
            pace_inputs = pace_tokenizer(response, return_tensors="pt", truncation=True, padding=True, max_length=128)
            pace_outputs = pace_model(**pace_inputs)
            pace_logits = pace_outputs.logits
            label_idx = torch.argmax(pace_logits, axis=1).item()
            # Classify: 0 -> "slow", 1 -> "fast"
            pace_label = "slow" if label_idx == 0 else "fast"
            return ("pace", pace_label)

        elif question in FOCUS_QUESTIONS:
            # Use focus model for people/task
            focus_inputs = focus_tokenizer(response, return_tensors="pt", truncation=True, padding=True, max_length=128)
            focus_outputs = focus_model(**focus_inputs)
            focus_logits = focus_outputs.logits
            label_idx = torch.argmax(focus_logits, axis=1).item()
            # Classify: 0 -> "task", 1 -> "people"
            focus_label = "task" if label_idx == 0 else "people"
            return ("focus", focus_label)

        else:
            # Fallback if a question doesn't match any known list
            return ("unknown", "N/A")

    def chatbot():
        st.title("DISC Assessment System")
        st.write("Answer 6 DISC-Personality Assessment Questions. Each answer must be at least 20 words.")

        # Initialize session state for indexing questions and storing responses
        if 'current_question' not in st.session_state:
            st.session_state.current_question = 0
            st.session_state.responses = []

        # If we haven't exhausted the 6 questions
        if st.session_state.current_question < len(all_questions):
            question = all_questions[st.session_state.current_question]
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
                        st.warning(
                            f"Your answer is too short. Please provide at least 20 words. Currently, you have {word_count} words."
                        )
                    elif not is_relevant(response, question):
                        st.warning("Your answer is not relevant to the question. Please answer again.")
                    else:
                        # Classify response using the question-based logic
                        model_type, label = classify_response(question, response)

                        # Store the results in session_state
                        st.session_state.responses.append({
                            "question": question,
                            "response": response,
                            "model_type": model_type,    # 'pace' or 'focus'
                            "label": label              # 'fast'/'slow' or 'people'/'task'
                        })

                        # Move to the next question
                        st.session_state.current_question += 1
        else:
            # All questions answered
            st.success("All questions answered!")
            st.write("Thank you very much for your time.")

            # Calculate the new DiSC Scores
            count_fast = sum(1 for r in st.session_state.responses if r["model_type"] == "pace" and r["label"] == "fast")
            count_slow = sum(1 for r in st.session_state.responses if r["model_type"] == "pace" and r["label"] == "slow")
            count_people = sum(1 for r in st.session_state.responses if r["model_type"] == "focus" and r["label"] == "people")
            count_task = sum(1 for r in st.session_state.responses if r["model_type"] == "focus" and r["label"] == "task")

            # Convert to fractions (since we have 3 questions each for pace and focus)
            fraction_fast = count_fast / 3
            fraction_slow = count_slow / 3
            fraction_people = count_people / 3
            fraction_task = count_task / 3

            # Compute D, I, S, C
            D = fraction_fast * fraction_task
            I = fraction_fast * fraction_people
            S = fraction_slow * fraction_people
            C = fraction_slow * fraction_task

            # Convert to percentage
            D_percent = D * 100
            I_percent = I * 100
            S_percent = S * 100
            C_percent = C * 100

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

            # Show result in table format
            df_data = []
            for idx, ans in enumerate(st.session_state.responses, start=1):
                df_data.append({
                    "Question Number": idx,
                    "Question": ans["question"],
                    "Response": ans["response"],
                    "Model Used": ans["model_type"],     # 'pace' or 'focus'
                    "Predicted Label": ans["label"]      # 'fast'/'slow' or 'people'/'task'
                })
            df = pd.DataFrame(df_data)
            st.table(df)

    chatbot()


# -----------------------------------------------------------------------------
# Page: Chatbot Chat with Candidate Results (with BigQuery "Chat with Data")
# -----------------------------------------------------------------------------
def chatbot_chat_with_candidate_result_page():
    st.markdown("<h2 style='text-align: center; color: #009688;'>Chat with Candidate Results</h2>", unsafe_allow_html=True)
    st.write("Let's start a conversation with the candidate.")

    ##############################################################
    # 1) Setup Gemini model + BigQuery client
    ##############################################################
    gemini_api_key = st.session_state.get("gemini_api_key")
    model = None

    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.Model("gemini-pro")  # Or your model variant
            st.success("Gemini API Key successfully configured.")
        except Exception as e:
            st.error(f"Error setting up the Gemini model: {e}")
    else:
        st.warning("Provide a valid Gemini API Key in HR Login to enable chat with data.")

    # Create BigQuery client if your credentials are set (e.g. via st.secrets)
    # Make sure your environment has: os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ...
    try:
        client = bigquery.Client()
        bigquery_ready = True
    except Exception as e:
        st.error(f"BigQuery client error: {e}")
        bigquery_ready = False

    ##############################################################
    # 2) Define table schema + prompts
    ##############################################################
    table_schema = """
    table_metadata = candidate_disc_table
    table_name: Candidate_DiSC,
    description: This table contains DiSC personality data for each candidate, along with basic info.
        name:
            data_type: String,
            description: Candidate's first name,
            example_value: Mike
            ,
        surname:
            data_type: String,
            description: Candidate's last name,
            example_value: Ross
            ,
        age:
            data_type: Integer,
            description: Candidate's age,
            example_value: 26
            ,
        gender:
            data_type: String,
            description: Candidate's gender,
            example_value: Male
            ,
        apply_position:
            data_type: String,
            description: Position for which the candidate is applying,
            example_value: Project Manager
            ,
        disc_result:
            data_type: String,
            description: Which DiSC type they most strongly align with,
            example_value: Dominance
            ,
        d_percentage:
            data_type: Float,
            description: Dominance percentage,
            example_value: 0.5
            ,
        i_percentage:
            data_type: Float,
            description: Influence percentage,
            example_value: 0.23
            ,
        s_percentage:
            data_type: Float,
            description: Steadiness percentage,
            example_value: 0.1
            ,
        c_percentage:
            data_type: Float,
            description: Conscientiousness percentage,
            example_value: 0.17
    """

    # Prompt 1: Generate BigQuery SQL from user text
    big_query_prompt = """
    You are a sophisticated BigQuery SQL query generator.
    Translate the following natural language request (human query) into a valid BigQuery syntax (SQL query).
    Consider the table schema provided.
    FROM always `your-project.your_dataset.your_table`
    Format the SQL Query result as JSON with 'big_query' as a key.

    ###
    Table Schema: {table_schema}
    Human Query: {query}
    SQL Query:
    """

    # Prompt 2: Summarize results
    response_prompt = """
    Summarize the DiSC personality data based on the user’s question and the query result. 
    Include relevant candidate info (like name, disc_result, percentages, apply_position, etc.) in your answer.

    ###
    Question: {user_query}
    Query result: {sql_result}
    Answer:
    """

    ##############################################################
    # 3) Helper: Run text-to-SQL chain, query BigQuery, summarize
    ##############################################################
    def run_disc_query(user_query: str):
        """
        1) Generate BigQuery SQL from user query using Gemini
        2) Execute the query
        3) Summarize results using Gemini
        """
        # A) Generate BigQuery SQL
        # We fill in big_query_prompt with table_schema + user_query
        prompt_for_sql = big_query_prompt.format(table_schema=table_schema, query=user_query)
        sql_gen_response = model.generate_content(prompt_for_sql).text
        
        # Attempt to parse the 'big_query' from JSON (the LLM is asked to format it)
        import json
        generated_sql = ""
        try:
            parsed = json.loads(sql_gen_response)
            generated_sql = parsed["big_query"]
        except:
            # If JSON parsing fails, fallback on the entire output
            generated_sql = sql_gen_response.strip()

        # B) Query BigQuery
        df_result = pd.DataFrame()
        if bigquery_ready and generated_sql:
            query_job = client.query(generated_sql)
            df_result = query_job.to_dataframe()

        # C) Summarize the query result
        # Convert DataFrame to a python dict so LLM can read it
        result_str = df_result.to_dict(orient="records")
        prompt_for_summary = response_prompt.format(user_query=user_query, sql_result=result_str)
        summary_response = model.generate_content(prompt_for_summary).text

        return summary_response, generated_sql

    ##############################################################
    # 4) Existing Chat UI logic
    ##############################################################
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing chat
    for role, message in st.session_state.chat_history:
        st.chat_message(role).markdown(message)

    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        st.session_state.chat_history.append(("user", user_input))
        st.chat_message("user").markdown(user_input)

        if model:
            try:
                # ------------------------------------------
                # "Chat with Data" step
                # ------------------------------------------
                # 1) We generate the text → SQL → result → summary
                summary_answer, generated_sql = run_disc_query(user_input)

                # 2) Add the answer to chat
                st.session_state.chat_history.append(("assistant", summary_answer))
                st.chat_message("assistant").markdown(summary_answer)

                with st.expander("See Generated SQL"):
                    st.code(generated_sql)

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Provide a valid Gemini API Key to start the conversation.")



# -----------------------------------------------------------------------------
# Page Navigation
# -----------------------------------------------------------------------------
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
