import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# 1. Load Environment Variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("GROQ_API_KEY") # CrewAI needs this env var
os.environ["OPENAI_MODEL_NAME"] = 'groq/llama-3.1-8b-instant'   # Groq ka model specify kiya

# --- Streamlit Page Setup ---
st.set_page_config(page_title="AI Agent Crew", layout="wide")
st.title("🤖 My AI Crew Dashboard")

topic = st.text_input("Enter Research Topic:", "AI Agent building platforms")

# --- Agent Definitions ---
researcher = Agent(
    role='Senior Market Researcher',
    goal='Analyze the latest trends and competitors in {topic}',
    backstory='You are an expert researcher with a keen eye for detail.',
    verbose=True,
    allow_delegation=False
)

writer = Agent(
    role='Content Strategist',
    goal='Write a compelling report based on the research',
    backstory='You are a master storyteller who simplifies complex data.',
    verbose=True,
    allow_delegation=False
)

# --- Task Definitions ---
task1 = Task(
    description='Conduct a deep dive into {topic} and identify key players.',
    expected_output='A summary of 5 key findings.',
    agent=researcher
)

task2 = Task(
    description='Create a blog post or report based on the research findings.',
    expected_output='A full report in markdown format.',
    agent=writer
)

# --- Execution Logic ---
if st.button("Run Crew Tasks"):
    with st.spinner('🚀 Agents kaam shuru kar rahe hain... (Check Terminal for logs)'):
        
        crew = Crew(
            agents=[researcher, writer],
            tasks=[task1, task2],
            process=Process.sequential,
            verbose=True
        )

        result = crew.kickoff(inputs={'topic': topic})

        st.success("✅ Mission Accomplished!")
        st.subheader("🏁 Final Report")
        st.markdown("---")
        st.markdown(result)
        
        st.download_button(
            label="Download Report",
            data=str(result),
            file_name="report.txt"
        )