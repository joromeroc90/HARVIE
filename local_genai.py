import asyncio
import json
import unicodedata
from datetime import datetime
import urllib.request

# ====================================================================
# 1. TYPES DEFINITION (Mocks google.genai.types)
# ====================================================================

class types:
    class PrebuiltVoiceConfig:
        def __init__(self, voice_name: str = None, **kwargs):
            self.voice_name = voice_name

    class VoiceConfig:
        def __init__(self, prebuilt_voice_config=None, **kwargs):
            self.prebuilt_voice_config = prebuilt_voice_config

    class SpeechConfig:
        def __init__(self, voice_config=None, **kwargs):
            self.voice_config = voice_config

    class SessionResumptionConfig:
        def __init__(self, **kwargs):
            pass

    class FunctionResponse:
        def __init__(self, id: str, name: str, response: dict):
            self.id = id
            self.name = name
            self.response = response

    class LiveConnectConfig:
        def __init__(
            self, 
            response_modalities: list = None, 
            system_instruction=None, 
            tools: list = None, 
            speech_config=None,
            output_audio_transcription=None,
            input_audio_transcription=None,
            session_resumption=None,
            **kwargs
        ):
            self.response_modalities = response_modalities or []
            self.system_instruction = system_instruction
            self.tools = tools or []
            self.speech_config = speech_config
            self.output_audio_transcription = output_audio_transcription
            self.input_audio_transcription = input_audio_transcription
            self.session_resumption = session_resumption

# ====================================================================
# 2. ASYNC SESSION HANDLER 
# ====================================================================

class _Transcription:
    def __init__(self, text: str = ""):
        self.text = text

class _ServerContent:
    def __init__(self, output_text: str = "", input_text: str = "", turn_complete: bool = False):
        self.output_transcription = _Transcription(output_text)
        self.input_transcription = _Transcription(input_text)
        self.turn_complete = turn_complete

class _FunctionCall:
    def __init__(self, name: str, args: dict, id: str = "local_call"):
        self.name = name
        self.args = args
        self.id = id

class _ToolCall:
    def __init__(self, function_calls: list):
        self.function_calls = function_calls

class _LiveResponse:
    def __init__(self, data: bytes = None, server_content=None, tool_call=None):
        self.data = data
        self.server_content = server_content
        self.tool_call = tool_call

class LocalLiveSession:
    def __init__(self, model: str, config: types.LiveConnectConfig):
        self.model = model
        self.config = config
        self._receive_queue = asyncio.Queue()
        self._history = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def receive(self):
        while True:
            response = await self._receive_queue.get()
            yield response

    async def send_client_content(self, turns: dict, turn_complete: bool = False):
        text_parts = [p.get("text", "") for p in (turns or {}).get("parts", [])
                      if isinstance(p, dict) and p.get("text")]
        user_text = "\n".join(text_parts).strip()
        if not user_text:
            return
        # Common voice command: answer locally and immediately, without
        # waiting for Ollama to generate a response.
        normalized = "".join(
            c for c in unicodedata.normalize("NFD", user_text.lower())
            if unicodedata.category(c) != "Mn"
        )
        if any(phrase in normalized for phrase in ("que hora es", "dime la hora", "hora actual")):
            answer = f"Son las {datetime.now().strftime('%H:%M')}, jefe."
            self._history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ])
            await self._receive_queue.put(_LiveResponse(
                server_content=_ServerContent(output_text=answer, turn_complete=True)
            ))
            return
        self._history.append({"role": "user", "content": user_text})
        system = getattr(self.config, "system_instruction", "") or ""
        messages = ([{"role": "system", "content": str(system)}]
                    if system else []) + self._history[-6:]

        def _ask_ollama():
            req = urllib.request.Request(
                "http://127.0.0.1:11434/api/chat",
                data=json.dumps({"model": "llama3.2", "messages": messages,
                                 "stream": False,
                                 "options": {"num_predict": 160, "temperature": 0.35}}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                return data.get("message", {}).get("content", "No recibí una respuesta.")

        try:
            answer = await asyncio.to_thread(_ask_ollama)
        except Exception as exc:
            answer = f"No pude conectar con el modelo local: {exc}"
        self._history.append({"role": "assistant", "content": answer})
        await self._receive_queue.put(_LiveResponse(
            server_content=_ServerContent(
                output_text=answer,
                turn_complete=True
            )
        ))

    async def send_realtime_input(self, media: dict):
        pass

    async def send_tool_response(self, function_responses: list):
        for fr in function_responses:
            print(f"[Local Tool Executed] {fr.name} -> {fr.response}")
        
        await self._receive_queue.put(_LiveResponse(
            server_content=_ServerContent(turn_complete=True)
        ))

# ====================================================================
# 3. CLIENT EXPORTS & OPERATIONAL MODELS
# ====================================================================

class _LiveClient:
    def connect(self, model: str, config: types.LiveConnectConfig):
        return LocalLiveSession(model=model, config=config)

class _AsyncClient:
    def __init__(self):
        self.live = _LiveClient()

class _Models:
    def generate_content(self, model: str, contents, **kwargs):
        # Operational connection to a local Ollama instance
        url = "http://127.0.0.1:11434/api/generate"
        
        payload = {
            "model": "llama3.2",
            "prompt": str(contents),
            "stream": False
        }
        
        try:
            req = urllib.request.Request(
                url, 
                data=json.dumps(payload).encode('utf-8'), 
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode('utf-8'))
                
                class LiveResponse:
                    def __init__(self, text):
                        self.text = text
                        
                return LiveResponse(text=result.get("response", "Search synthesis returned empty."))
        except Exception as e:
            class ErrorResponse:
                def __init__(self, err_msg):
                    self.text = err_msg
            return ErrorResponse(err_msg=f"WebSearch AI Synthesis failed. Is the local LLM running? Error: {e}")

class Client:
    def __init__(self, api_key: str = None, http_options: dict = None):
        self.aio = _AsyncClient()
        self.models = _Models()
