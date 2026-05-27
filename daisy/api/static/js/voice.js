/* D.A.I.S.Y. v2 — Voice Session
 *
 * Manages the lifecycle of a remote voice session:
 *   1. getUserMedia → AudioContext → AudioWorklet → Int16 frames
 *   2. Sends mic audio as binary WebSocket frames (0x00 prefix)
 *   3. Coordinates with DaisyAudioPlayer for TTS playback
 */

export class VoiceSession {
    constructor(client, audioPlayer) {
        this._client = client;
        this._player = audioPlayer;
        this._ctx = null;
        this._worklet = null;
        this._stream = null;
        this.active = false;
    }

    async start() {
        if (this.active) return;

        // 1. Request microphone
        this._stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            },
        });

        // 2. Create AudioContext at 16 kHz (mic native rate)
        this._ctx = new AudioContext({ sampleRate: 16000 });
        const source = this._ctx.createMediaStreamSource(this._stream);

        // 3. Load AudioWorklet processor
        await this._ctx.audioWorklet.addModule('/js/audio-processor.js');
        this._worklet = new AudioWorkletNode(this._ctx, 'daisy-mic-processor');

        // 4. Receive processed PCM from worklet, send to server
        this._worklet.port.onmessage = (event) => {
            const int16Buffer = event.data; // Int16Array buffer
            // Prepend 0x00 marker byte
            const frame = new Uint8Array(int16Buffer.byteLength + 1);
            frame[0] = 0x00; // mic audio marker
            frame.set(new Uint8Array(int16Buffer), 1);
            this._client.sendBinary(frame.buffer);
        };

        // 5. Connect graph (worklet doesn't need destination — we capture, not monitor)
        source.connect(this._worklet);

        // 6. Notify server
        this._client.send('voice_start');
        this.active = true;
    }

    stop() {
        if (!this.active) return;

        if (this._worklet) {
            this._worklet.disconnect();
            this._worklet.port.onmessage = null;
            this._worklet = null;
        }
        if (this._ctx) {
            this._ctx.close();
            this._ctx = null;
        }
        if (this._stream) {
            this._stream.getTracks().forEach(t => t.stop());
            this._stream = null;
        }

        this._client.send('voice_stop');
        this.active = false;
    }
}
