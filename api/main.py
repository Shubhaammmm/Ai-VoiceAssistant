import time
import threading
import numpy as np
import whisper
import sounddevice as sd
import webrtcvad
from queue import Queue
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from api.tts import speak_text
from api.config import Config
from api.rag import retrieve_context


console = Console()
stt = whisper.load_model(Config.WHISPER_MODEL)

# Enhanced prompt template with personality
template = """
You are Heckx, a witty and helpful AI assistant created by bobo. You provide concise, accurate answers with a touch of humor.
Use the relevant context below to answer questions accurately and clearly.
Keep responses under 30 words unless asked for more detail.

Conversation history:
{history}

Relevant context:
{context}

User's input: {input}

Your response:
"""

PROMPT = PromptTemplate(input_variables=["history", "input", "context"], template=template)
llm = ChatOllama(model=Config.OLLAMA_MODEL)
memory = ConversationBufferMemory(
    ai_prefix="Heckx:",
    human_prefix="You:",
    return_messages=False,
    k=Config.CONVERSATION_HISTORY_LIMIT
)

class VoiceAssistant:
    def __init__(self):
        self.console = console
        self.data_queue = Queue()
        self.stop_event = threading.Event()

    # def record_audio(self):
    #     vad = webrtcvad.Vad(1)  # Aggressiveness: 0–3
    #     ring_buffer = []
    #     triggered = False
    #     voiced_frames = []

    #     def callback(indata, frames, time_info, status):
    #         nonlocal triggered, ring_buffer, voiced_frames

    #         if status:
    #             self.console.print(f"[red]Audio error: {status}")

    #         frame = bytes(indata)
    #         is_speech = vad.is_speech(frame, Config.SAMPLE_RATE)

    #         if not triggered:
    #             ring_buffer.append(frame)
    #             if len(ring_buffer) > 10:
    #                 ring_buffer.pop(0)
    #             if is_speech:
    #                 triggered = True
    #                 voiced_frames.extend(ring_buffer)
    #                 ring_buffer = []
    #         else:
    #             voiced_frames.append(frame)
    #             if not is_speech:
    #                 ring_buffer.append(frame)
    #                 if len(ring_buffer) > 20:
    #                     self.data_queue.put(b"".join(voiced_frames))
    #                     triggered = False
    #                     voiced_frames = []
    #                     ring_buffer = []

    #     with sd.RawInputStream(
    #         samplerate=Config.SAMPLE_RATE,
    #         blocksize=Config.FRAME_SIZE,
    #         dtype=Config.DTYPE,
    #         channels=Config.CHANNELS,
    #         callback=callback
    #     ):
    #         while not self.stop_event.is_set():
    #             time.sleep(0.1)

    def record_audio(self):
        """Captures audio and puts it in the queue."""
        def callback(indata, frames, time_info, status):
            if status:
                self.console.print(f"[red]Audio Error: {status}")
            if self.is_recording:
                self.data_queue.put(bytes(indata))

        with sd.RawInputStream(
            samplerate=Config.SAMPLE_RATE,
            dtype=Config.DTYPE,
            channels=Config.CHANNELS,
            callback=callback
        ):
            while not self.stop_event.is_set():
                time.sleep(0.1)


    def transcribe(self, audio_np: np.ndarray) -> str:
        result = stt.transcribe(audio_np, fp16=False)
        return result["text"].strip()

    def get_response(self, user_input: str) -> str:
        history = memory.load_memory_variables({})["history"]
        context = retrieve_context(user_input)

        # Fill the prompt manually
        full_prompt = PROMPT.format(
            history=history,
            input=user_input,
            context=context
        )
        # Run LLM
        response = llm.invoke(full_prompt)

        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)

        # Save only the string to memory
        memory.save_context({"input": user_input}, {"output": response_text})

        return response_text

    def display_welcome(self):
        welcome_text = Text.assemble(
            ("Heckx is ready and listening for speech.\n", "cyan bold"),
            ("Speak naturally. It will auto-respond after you're done.\n", "white"),
            ("Press Ctrl+C to stop. Let's chat!", "cyan")
        )
        self.console.print(Panel(welcome_text, title="Heckx (VAD Mode)", border_style="blue"))

    # def run(self):
    #     self.display_welcome()
    #     try:
    #         while True:
    #             self.console.print("[green]Listening for speech...[/green]")

    #             self.data_queue = Queue()
    #             self.stop_event.clear()

    #             recording_thread = threading.Thread(target=self.record_audio)
    #             recording_thread.start()

    #             while self.data_queue.empty():
    #                 time.sleep(0.1)

    #             self.stop_event.set()
    #             recording_thread.join()

    #             audio_data = self.data_queue.get()
    #             audio_np = (
    #                 np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    #             )

    #             if audio_np.size > 0:
    #                 with self.console.status("[blue]Transcribing...", spinner="dots"):
    #                     text = self.transcribe(audio_np)

    #                 self.console.print(Panel(f"You: {text}", title="Your Input", border_style="yellow"))

    #                 with self.console.status("[blue]Thinking...", spinner="moon"):
    #                     response = self.get_response(text)

    #                 self.console.print(Panel(f"Heckx: {response}", title="Heckx's Response", border_style="cyan"))
    #                 speak_text(response)

    #     except KeyboardInterrupt:
    #         self.console.print("\n[red]Shutting down gracefully...[/red]")
    #         self.stop_event.set()
    #         self.console.print(Panel(
    #             "Thanks for chatting! Come back anytime!",
    #             title="Goodbye",
    #             border_style="blue"
    #         ))

    def run(self):
        self.display_welcome()

        try:
            while True:
                self.console.input("[green]Press Enter to start speaking...[/green]")

                self.data_queue = Queue()
                self.stop_event.clear()
                self.is_recording = True

                recording_thread = threading.Thread(target=self.record_audio)
                recording_thread.start()

                self.console.input("[yellow]Speaking... Press Enter to stop[/yellow]")
                self.is_recording = False
                self.stop_event.set()
                recording_thread.join()

                audio_data = b"".join(list(self.data_queue.queue))
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                )

                if audio_np.size > 0:
                    with self.console.status("[blue]Transcribing...", spinner="dots"):
                        text = self.transcribe(audio_np)

                    self.console.print(Panel(f"You: {text}", title="Your Input", border_style="yellow"))

                    with self.console.status("[blue]Thinking...", spinner="moon"):
                        response = self.get_response(text)

                    self.console.print(Panel(f"Heckx: {response}", title="Heckx's Response", border_style="cyan"))
                    speak_text(response)

                else:
                    self.console.print(Panel(
                        "No audio detected. Check your microphone!",
                        title="Error",
                        border_style="red"
                    ))

        except KeyboardInterrupt:
            self.console.print("\n[red]Shutting down gracefully...[/red]")
            self.stop_event.set()
            self.console.print(Panel(
                "Thanks for chatting! Come back anytime!",
                title="Goodbye",
                border_style="blue"
            ))


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
