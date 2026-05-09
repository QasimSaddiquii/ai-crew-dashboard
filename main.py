import streamlit as st
import os
from crewai import Agent, Task, Crew, Process

# --- 1. API Key Configuration (Streamlit Secrets) ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["OTEL_SDK_DISABLED"] = "true" 
    os.environ["OPENAI_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["OPENAI_MODEL_NAME"] = 'groq/llama-3.1-8b-instant'
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1"

# --- 2. Streamlit Page Setup ---
st.set_page_config(page_title="AI Agent Crew", layout="wide", page_icon="🤖")
st.title("🤖 My AI Crew Dashboard")

topic = st.text_input("Enter Research Topic:", "Future of AI Agents")

# --- 3. Agent Definitions ---
researcher = Agent(
    role='Expert Researcher',
    goal=f'Conduct a deep dive research on {topic}',
    backstory='You are a world-class researcher known for finding trends.',
    llm='groq/llama-3.1-8b-instant', # Direct LLM string
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role='Content Strategist',
    goal='Write a compelling report based on the research',
    backstory='You are a master storyteller.',
    llm='groq/llama-3.1-8b-instant', # Direct LLM string
    allow_delegation=False,
    verbose=True
)

# --- 4. Task Definitions ---
task1 = Task(
    description=f'Identify 5 key trends in {topic}.',
    expected_output='A detailed summary of 5 key findings.',
    agent=researcher
)

task2 = Task(
    description='Create a professional blog post based on the findings.',
    expected_output='A full blog post in markdown format.',
    agent=writer
)

# --- 5. Execution Logic ---
if st.button("🚀 Run Crew Tasks"):
    if "GROQ_API_KEY" not in st.secrets:
        st.error("❌ Error: API Key missing in Secrets!")
    else:
        with st.spinner('Agents are working...'):
            try:
                crew = Crew(
                    agents=[researcher, writer],
                    tasks=[task1, task2],
                    process=Process.sequential,
                    verbose=True
                )
                result = crew.kickoff(inputs={'topic': topic})
                st.success("✅ Mission Accomplished!")
                st.markdown(str(result))
            except Exception as e:
                st.error(f"An error occurred: {e}")