import logging
import json
import os
from typing import Annotated, Literal, Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)

from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")



CONTENT_FILE = "day4_tutor_content.json"

DEFAULT_CONTENT = [
    {
        "id": "variables",
        "title": "Variables",
        "summary": "Variables are like labeled containers that store information in your program. Think of them as boxes with name tags - you can put data in them, check what's inside, or change the contents. They're useful because they let you reuse values throughout your code without typing them repeatedly, and they make your code more flexible and easier to update.",
        "sample_question": "What is a variable and why is it useful in programming?"
    },
    {
        "id": "loops",
        "title": "Loops",
        "summary": "Loops are programming structures that repeat actions multiple times automatically. Instead of writing the same code over and over, you use a loop to tell the computer 'do this 10 times' or 'keep doing this until a condition is met.' The two main types are 'for loops' which repeat a specific number of times, and 'while loops' which repeat as long as a condition stays true.",
        "sample_question": "Explain the difference between a for loop and a while loop. When would you use each one?"
    },
    {
        "id": "functions",
        "title": "Functions",
        "summary": "Functions are reusable blocks of code that perform a specific task. Like a recipe, they take inputs (called parameters), do something with them, and often return a result. Functions help organize code, avoid repetition, and make programs easier to understand and maintain. You can think of them as mini-programs within your main program.",
        "sample_question": "What is a function and what are the benefits of using functions in your code?"
    },
    {
        "id": "conditionals",
        "title": "Conditionals",
        "summary": "Conditionals are decision-making structures in code, using if-then-else logic. They let your program choose different paths based on whether conditions are true or false. For example, 'if the user is logged in, show the dashboard, else show the login page.' This makes programs dynamic and responsive to different situations.",
        "sample_question": "How do conditionals work and why are they important for making programs interactive?"
    },
    {
        "id": "data_types",
        "title": "Data Types",
        "summary": "Data types define what kind of information a variable can hold - like numbers, text, true/false values, or collections of items. Each type has different capabilities: you can do math with numbers, combine text strings, or check boolean true/false conditions. Understanding data types helps prevent errors and lets you use the right operations for each kind of data.",
        "sample_question": "Name three common data types and explain what each one is used for."
    }
]

def load_content():
    """📖 Loads or creates the tutor content JSON file"""
    try:
        path = os.path.join(os.path.dirname(__file__), CONTENT_FILE)
        
        if not os.path.exists(path):
            print(f"⚠️ {CONTENT_FILE} not found. Generating content...")
            with open(path, "w", encoding='utf-8') as f:
                json.dump(DEFAULT_CONTENT, f, indent=2)
            print("✅ Content file created successfully.")
            
        with open(path, "r", encoding='utf-8') as f:
            data = json.load(f)
            return data
            
    except Exception as e:
        print(f"⚠️ Error managing content file: {e}")
        return DEFAULT_CONTENT

COURSE_CONTENT = load_content()



@dataclass
class TutorState:
    """🧠 Tracks the current learning context"""
    current_topic_id: str | None = None
    current_topic_data: dict | None = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"
    
    def set_topic(self, topic_id: str):
        """Set the current topic by ID"""
        topic = next((item for item in COURSE_CONTENT if item["id"] == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            return True
        return False

@dataclass
class Userdata:
    """User session data"""
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None 



@function_tool
async def select_topic(
    ctx: RunContext[Userdata], 
    topic_id: Annotated[str, Field(description="The ID of the topic to study (e.g., 'variables', 'loops', 'functions')")]
) -> str:
    """📚 Selects a topic to study from the available list."""
    state = ctx.userdata.tutor_state
    success = state.set_topic(topic_id.lower())
    
    if success:
        return f"Topic set to '{state.current_topic_data['title']}'. Now ask the user which mode they'd like: 'Learn' to have it explained, 'Quiz' to be tested, or 'Teach Back' to explain it themselves."
    else:
        available = ", ".join([t["id"] for t in COURSE_CONTENT])
        return f"Topic '{topic_id}' not found. Available topics: {available}"

@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata], 
    mode: Annotated[str, Field(description="The mode to switch to: 'learn', 'quiz', or 'teach_back'")]
) -> str:
    """🔄 Switches the interaction mode and updates the agent's voice/persona."""
    
    state = ctx.userdata.tutor_state
    
    if not state.current_topic_data:
        return "Please select a topic first before choosing a learning mode."
    
    state.mode = mode.lower()
    
    agent_session = ctx.userdata.agent_session 
    
    if agent_session:
        if state.mode == "learn":
            agent_session.tts.update_options(voice="en-US-matthew", style="Conversation")
            instruction = f"Switched to LEARN mode. Now explain this concept clearly: {state.current_topic_data['summary']} Use simple terms and concrete examples."
            
        elif state.mode == "quiz":
            agent_session.tts.update_options(voice="en-US-alicia", style="Conversation")
            instruction = f"Switched to QUIZ mode. Ask this question: {state.current_topic_data['sample_question']} After they answer, provide feedback on their response."
            
        elif state.mode == "teach_back":
            agent_session.tts.update_options(voice="en-US-ken", style="Conversation")
            instruction = f"Switched to TEACH-BACK mode. Ask the user to explain '{state.current_topic_data['title']}' to you as if you're a complete beginner. Listen carefully to their explanation."
        else:
            return "Invalid mode. Use 'learn', 'quiz', or 'teach_back'."
    else:
        instruction = "Voice switch failed (session not found)."

    print(f"🔄 MODE SWITCH: {state.mode.upper()} | Topic: {state.current_topic_data['title']}")
    return instruction

@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="The user's explanation of the concept in teach-back mode")]
) -> str:
    """📝 Evaluates the user's explanation when they teach the concept back."""
    
    state = ctx.userdata.tutor_state
    
    if not state.current_topic_data:
        return "No topic selected to evaluate."
    
    correct_summary = state.current_topic_data['summary']
    
    print(f"📝 EVALUATING: {user_explanation[:100]}...")
    
    return f"""Analyze the user's explanation against this correct summary: '{correct_summary}'.

Provide feedback in this format:
1. Score their explanation (1-10) for accuracy and clarity
2. What they explained well (be specific)
3. What was missing or could be clearer
4. One concrete suggestion for improvement

Be encouraging and constructive!"""

@function_tool
async def list_available_topics(
    ctx: RunContext[Userdata]
) -> str:
    """📋 Lists all available topics the user can study."""
    topics = [f"• {t['id']}: {t['title']}" for t in COURSE_CONTENT]
    return "Available topics:\n" + "\n".join(topics)



class ActiveRecallCoach(Agent):
    def __init__(self):
        # Generate topic list for instructions
        topic_list = ", ".join([f"{t['id']}" for t in COURSE_CONTENT])
        
        super().__init__(
            instructions=f"""You are an Active Recall Coach that helps users master concepts through three proven learning techniques.

📚 **AVAILABLE TOPICS:** {topic_list}

🎯 **THREE LEARNING MODES:**

1. **LEARN Mode **
   - You're a patient teacher using the Feynman Technique
   - Explain concepts clearly with concrete examples
   - Break down complex ideas into digestible pieces

2. **QUIZ Mode **
   - You're an engaging quiz master testing comprehension
   - Ask thoughtful questions to check understanding
   - Provide constructive feedback on answers

3. **TEACH-BACK Mode **
   - You're a curious learner who wants to understand
   - Ask the user to explain the concept to you
   - Listen carefully, then evaluate their explanation

⚙️ **HOW TO INTERACT:**

1. First, ask what topic they want to study (use `list_available_topics` if they're unsure)
2. Use `select_topic` when they choose a topic
3. Ask which mode they prefer: learn, quiz, or teach back
4. Use `set_learning_mode` immediately when they indicate their choice
5. In teach-back mode, after hearing their full explanation, use `evaluate_teaching` to give detailed feedback
6. Users can switch topics or modes anytime - just use the appropriate tool

💡 **TIPS:**
- Keep responses conversational and encouraging
- In teach-back mode, really listen before evaluating
- Celebrate progress and effort
- Make learning feel supportive, not intimidating""",
            tools=[select_topic, set_learning_mode, evaluate_teaching, list_available_topics],
        )

def prewarm(proc: JobProcess):
    """Preload VAD model"""
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """Main entry point for the agent"""
    ctx.log_context_fields = {"room": ctx.room.name}


    
    userdata = Userdata(tutor_state=TutorState())

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",   
            style="Conversation",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )
    
    userdata.agent_session = session
    
    await session.start(
        agent=ActiveRecallCoach(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
