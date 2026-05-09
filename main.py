import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM

# CrewAI ko batana ke hum Groq use kar rahe hain
os.environ["LITELLM_LOG"] = "DEBUG" 

# Model ko sahi tarah configure karein
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
    my_llm = LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=api_key
    )

# --- Streamlit Setup ---
st.set_page_config(page_title="AI Agent Crew", layout="wide")
st.title("🤖 My AI Crew Dashboard")

topic = st.text_input("Enter Research Topic:", "Future of AI")

# --- Agent Definitions ---
researcher = Agent(
    role='Expert Researcher',
    goal=f'Research about {topic}',
    backstory='You are a tech researcher.',
    llm=my_llm, # <--- Direct model object use kiya
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role='Content Strategist',
    goal='Write a report',
    backstory='You are a master storyteller.',
    llm=my_llm, # <--- Direct model object use kiya
    allow_delegation=False,
    verbose=True
)

# --- Tasks ---
task1 = Task(description=f'Find 3 facts about {topic}', expected_output='3 points', agent=researcher)
task2 = Task(description='Write a summary', expected_output='A summary', agent=writer)

# --- Run ---
if st.button("🚀 Run Crew"):
    with st.spinner('Wait...'):
        crew = Crew(agents=[researcher, writer], tasks=[task1, task2], verbose=True)
        result = crew.kickoff(inputs={'topic': topic})
        st.markdown(str(result))