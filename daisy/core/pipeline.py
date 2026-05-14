import asyncio
from daisy.audio.input_stream import AudioInputStream
from daisy.audio.output_stream import AudioOutputStream
from daisy.vad.silero_vad import SileroVAD
from daisy.stt.faster_whisper_stt import FasterWhisperSTT
from daisy.llm.groq_client import GroqClient
from daisy.llm.sentence_splitter import SentenceSplitter
from daisy.tts.kokoro_tts import KokoroTTS


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.audio_in = AudioInputStream(config)
        self.audio_out = AudioOutputStream(config)
        self.vad = SileroVAD(config)
        self.stt = FasterWhisperSTT(config)
        self.llm = GroqClient(config)
        self.tts = KokoroTTS(config)

    async def start(self):
        await self.audio_in.start()
        self.audio_out.start()

    async def stop(self):
        self.audio_out.stop()
        await self.audio_in.stop()

    async def run_turn(self):
        import time
        t_start = time.time()

        t0 = time.time()
        audio_buffer = await self.vad.listen(self.audio_in)
        t1 = time.time()
        print(f"[timing] VAD listen: {t1-t0:.2f}s", file=__import__("sys").stderr)

        text = await self.stt.transcribe(audio_buffer)
        t2 = time.time()
        print(f"[timing] STT: {t2-t1:.2f}s | text: \"{text}\"", file=__import__("sys").stderr)
        if not text:
            return

        print(f"You: {text}")

        sentence_queue = asyncio.Queue()
        splitter = SentenceSplitter()

        async def tts_worker():
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break
                audio = self.tts.synthesize(sentence)
                self.audio_out.play(audio)

        async def llm_worker():
            async for token in self.llm.stream_tokens(text):
                sentence = splitter.process_token(token)
                if sentence:
                    await sentence_queue.put(sentence)
            remaining = splitter.flush()
            if remaining:
                await sentence_queue.put(remaining)
            await sentence_queue.put(None)

        tts_task = asyncio.create_task(tts_worker())
        llm_task = asyncio.create_task(llm_worker())

        await llm_task
        t3 = time.time()
        print(f"[timing] LLM stream done: {t3-t2:.2f}s", file=__import__("sys").stderr)

        await tts_task
        t4 = time.time()
        print(f"[timing] TTS synth done: {t4-t3:.2f}s", file=__import__("sys").stderr)

        await self.audio_out.wait_until_done()
        t5 = time.time()
        print(f"[timing] Playback done: {t5-t4:.2f}s", file=__import__("sys").stderr)
        print(f"[timing] Total turn: {t5-t_start:.2f}s", file=__import__("sys").stderr)
