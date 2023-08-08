import streamlit as st
import os
from lc_functions import load_data, split_text, initialize_llm, generate_questions, create_retrieval_qa_chain

st.title("Smart Study Buddy")

if 'questions' not in st.session_state:
    st.session_state['questions'] = 'empty'
    st.session_state['seperated_question_list'] = 'empty'
    st.session_state['questions_to_answers'] = 'empty'
    st.session_state['submitted'] = 'empty'

# Get the open AI API Key

os.environ["OPENAI_API_KEY"] = st.text_input(label="OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key")

# File uploader

uploaded_file = st.file_uploader(label="Upload study material", type=['pdf'])

if uploaded_file:
    # Load data from pdf
    text_from_pdf = load_data(uploaded_file)

    # Split the text for question gen
    documents_for_question_gen = split_text(text=text_from_pdf, chunk_size=10000, chunk_overlap=200)

    # Split the text for question answering
    documents_for_question_answer = split_text(text=text_from_pdf, chunk_size=1000, chunk_overlap=100)

    st.write("Number of documents for question generation: ", len(documents_for_question_gen))
    st.write("Number of documents for question answering: ", len(documents_for_question_answer))

    # Init llm for question generation
    llm_question_gen = initialize_llm(model="gpt-3.5-turbo-16k", temperature=0.4)

    # Init llm for question answering
    llm_question_answer = initialize_llm(model="gpt-3.5-turbo-16k", temperature=0.1)

    if st.session_state['questions'] == 'empty':
        with st.spinner("Generating questions..."):
            st.session_state['questions'] = generate_questions(llm=llm_question_answer, chain_type="refine", documents=documents_for_question_gen)

    if st.session_state['questions'] != 'empty':
        st.info(st.session_state['questions'])

        st.session_state['questions_list'] = st.session_state['questions'].split('\n')

        with st.form(key='my_form'):
            st.session_state['questions_to_answer'] = st.multiselect(label="Select questions to answer", options=st.session_state['questions_list'])

            submitted = st.form_submit_button("Generate Answer")

            if submitted:
                st.session_state['submitted'] = True

        if st.session_state['submitted']:
            with st.spinner("Generating answers..."):
                generate_answer_chain = create_retrieval_qa_chain(documents=documents_for_question_answer, llm=llm_question_answer)

                for question in st.session_state['questions_to_answer']:

                    answer = generate_answer_chain.run(question)

                    st.write(f"Question: {question}")
                    st.info(f"Answer: {answer}")


