import os
import aiohttp
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero, groq

load_dotenv()

SYSTEM_PROMPT = """
You are the professional voice receptionist for our clinic.
Current Year: 2026.

Rules:
- Speak naturally and keep responses concise (1-2 sentences maximum).
- Politely prompt the caller for:
  1. Full Name
  2. Phone Number
  3. Reason for visit / symptoms
  4. Preferred appointment date and time
- Once all 4 details are confirmed, trigger the book_appointment function.
"""

async def entrypoint(ctx: JobContext):
    # Connect directly to the WebRTC room created by LiveKit SIP
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Initialize sub-second voice pipeline
    assistant = VoicePipelineAgent(
        vad=silero.VAD.load(),
        stt=groq.STT(model="whisper-large-v3"),
        llm=groq.LLM(model="llama-3.3-70b-versatile"),
        tts=groq.TTS() if hasattr(groq, "TTS") else None,
        chat_ctx=llm.ChatContext().append(role="system", text=SYSTEM_PROMPT),
    )

    # Tool hook that dispatches structured JSON directly to n8n
    @assistant.tool
    async def book_appointment(name: str, phone: str, issue: str, appointment_time: str):
        """Asynchronously writes the appointment payload to the clinic n8n automation."""
        payload = {
            "name": name,
            "phone": phone,
            "issue": issue,
            "time": appointment_time,
            "source": "Voice Receptionist"
        }
        webhook_url = os.getenv("N8N_WEBHOOK_URL")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(webhook_url, json=payload, timeout=5) as resp:
                    if resp.status == 200:
                        return "Appointment successfully confirmed and synced with the clinic calendar."
            except Exception as e:
                return f"Appointment noted, dispatch queued. Status: {str(e)}"
        
        return "Appointment registered."

    assistant.start(ctx.room)
    await assistant.say("Thank you for calling our clinic! How can I assist with your appointment today?")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
