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
        await self.audio_in.stop()
        self.audio_out.stop()

    async def run_turn(self):
        audio_buffer = await self.vad.listen(self.audio_in)
        text = await self.stt.transcribe(audio_buffer)
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
        await tts_task

        await self.audio_out.wait_until_done()
