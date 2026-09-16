import os
import aiohttp
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import silero, groq

load_dotenv()

SYSTEM_PROMPT = """
You are the professional voice receptionist for our clinic.
Current Year: 2026.
Keep answers spoken, natural, and concise (1 to 2 sentences max).
Politely ask for:
1. Patient's full name
2. Contact phone number
3. Reason for visit / problem
4. Desired appointment date and time

Once all details are gathered, call the book_appointment function.
"""

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=groq.STT(model="whisper-large-v3"),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=groq.TTS() if hasattr(groq, "TTS") else None,
        chat_ctx=llm.ChatContext().append(role="system", text=SYSTEM_PROMPT),
    )

    @assistant.tool
    async def book_appointment(name: str, phone: str, issue: str, appointment_time: str):
        """Pushes booking directly into the clinic n8n workflow."""
        payload = {
            "name": name,
            "phone": phone,
            "issue": issue,
            "time": appointment_time
        }
        async with aiohttp.ClientSession() as session:
            await session.post(os.getenv("N8N_WEBHOOK_URL"), json=payload)
        return "The appointment has been confirmed and added to our schedule."

    assistant.start(ctx.room)
    await assistant.say("Thank you for calling our clinic! How can I assist with your appointment today?")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
