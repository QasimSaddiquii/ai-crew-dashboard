import streamlit as st
import os
from crewai import Agent, Task, Crew, Process

# --- 1. API Key Configuration (Streamlit Secrets) ---
# Hum Groq ki key ko OPENAI_API_KEY ke variable mein daal rahe hain 
# taake CrewAI ka default behavior sahi chale.
if "GROQ_API_KEY" in st.secrets:
    os.environ["OTEL_SDK_DISABLED"] = "true" 
    os.environ["OPENAI_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["OPENAI_MODEL_NAME"] = 'groq/llama-3.1-8b-instant'
    os.environ["OPENAI_API_BASE"] = "https://api.groq.com/openai/v1" # Groq API endpoint

# --- 2. Streamlit Page Setup ---
st.set_page_config(page_title="AI Agent Crew", layout="wide", page_icon="🤖")
st.title("🤖 My AI Crew Dashboard")
st.markdown("Build your research and content team with Groq & CrewAI.")

topic = st.text_input("Enter Research Topic:", "Future of AI Agents")

# --- 3. Agent Definitions ---
# Note: Humne llm specify kiya hai taake ye Groq use kare
researcher = Agent(
    role='Expert Researcher',
    goal=f'Conduct a deep dive research on {topic}',
    backstory='You are a world-class researcher known for finding hidden gems in technology trends.',
    llm='groq/llama-3.1-8b-instant',
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role='Content Strategist',
    goal='Write a compelling report based on the research',
    backstory='You are a master storyteller who simplifies complex data for the general public.',
    llm='groq/llama-3.1-8b-instant',
    allow_delegation=False,
    verbose=True
)

# --- 4. Task Definitions ---
task1 = Task(
    description=f'Conduct a deep dive into {topic} and identify 5 key trends.',
    expected_output='A detailed summary of 5 key findings in bullet points.',
    agent=researcher
)

task2 = Task(
    description='Create a professional blog post based on the research findings.',
    expected_output='A full blog post in markdown format with a catchy title.',
    agent=writer
)

# --- 5. Execution Logic ---
if st.button("🚀 Run Crew Tasks"):
    if not st.secrets.get("GROQ_API_KEY"):
        st.error("❌ Error: GROQ_API_KEY is missing in Streamlit Secrets!")
    else:
        with st.spinner('Agents are working... Please wait.'):
            try:
                # Crew Setup
                crew = Crew(
                    agents=[researcher, writer],
                    tasks=[task1, task2],
                    process=Process.sequential,
                    verbose=True
                )

                # Execute
                result = crew.kickoff(inputs={'topic': topic})

                # Display Results
                st.success("✅ Mission Accomplished!")
                st.subheader("🏁 Final Output")
                st.markdown("---")
                st.markdown(str(result))
                
                # Download Option
                st.download_button(
                    label="📥 Download Report",
                    data=str(result),
                    file_name="ai_report.md",
                    mime="text/markdown"
                )
            except Exception as e:
                st.error(f"An error occurred: {e}")