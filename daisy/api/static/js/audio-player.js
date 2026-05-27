/* D.A.I.S.Y. v2 — Audio Player
 *
 * Receives PCM16 audio chunks from the server, plays them through
 * the Web Audio API, and exposes an AnalyserNode so the orb can
 * read real-time frequency data for audio-reactive animation.
 */

export class DaisyAudioPlayer {
    constructor() {
        this._ctx = null;
        this._analyser = null;
        this._queue = [];
        this._playing = false;
        this._finishedCallback = null;
        this._sampleRate = 24000; // Kokoro TTS native rate
    }

    // --- Public ---

    /** Enqueue a chunk of Int16 PCM audio for playback. */
    enqueue(pcmInt16) {
        if (!pcmInt16 || pcmInt16.length === 0) return;
        // Convert Int16 → Float32
        const float32 = new Float32Array(pcmInt16.length);
        for (let i = 0; i < pcmInt16.length; i++) {
            float32[i] = pcmInt16[i] / 32768.0;
        }
        this._queue.push(float32);
        if (!this._playing) this._playNext();
    }

    /** Hard stop — clears queue and kills playback immediately. */
    stop() {
        this._queue.length = 0;
        if (this._ctx) {
            this._ctx.close();
            this._ctx = null;
            this._analyser = null;
        }
        this._playing = false;
    }

    /** Clear pending audio without killing current playback. */
    clear() {
        this._queue.length = 0;
    }

    /** Returns the AnalyserNode for orb visualisation, or null. */
    getAnalyser() {
        return this._analyser;
    }

    /** Called when all queued audio finishes playing. */
    onFinished(cb) {
        this._finishedCallback = cb;
    }

    /** Set sample rate before first enqueue (default 24000 for Kokoro). */
    setSampleRate(rate) {
        this._sampleRate = rate;
    }

    // --- Internal ---

    _ensureContext() {
        if (!this._ctx || this._ctx.state === 'closed') {
            this._ctx = new AudioContext({ sampleRate: this._sampleRate });
            this._analyser = this._ctx.createAnalyser();
            this._analyser.fftSize = 256;
            this._analyser.smoothingTimeConstant = 0.8;
            this._analyser.connect(this._ctx.destination);
        }
        if (this._ctx.state === 'suspended') {
            this._ctx.resume();
        }
    }

    _playNext() {
        if (this._queue.length === 0) {
            this._playing = false;
            if (this._finishedCallback) this._finishedCallback();
            return;
        }

        this._playing = true;
        this._ensureContext();

        const float32 = this._queue.shift();
        const buffer = this._ctx.createBuffer(1, float32.length, this._sampleRate);
        buffer.getChannelData(0).set(float32);

        const source = this._ctx.createBufferSource();
        source.buffer = buffer;
        source.onended = () => this._playNext();

        // Audio graph: source → analyser → speakers
        source.connect(this._analyser);
        // analyser is already connected to destination in _ensureContext

        source.start();
    }
}
