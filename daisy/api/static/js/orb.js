/* D.A.I.S.Y. v2 — JARVIS Particle Orb
 *
 * 2000-particle cloud with physics-based motion, audio-reactive forces,
 * connection lines, bloom post-processing, and smooth state transitions.
 *
 * Audio drives particle radial expansion (envelope now, AnalyserNode in Phase F8).
 * State changes lerp target radius, speed, color, line opacity, and particle size.
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

// --- State parameter targets ---
const STATE_PARAMS = {
    idle: {
        radius: 22, speed: 0.3, brightness: 0.6,
        color: 0x4ca8e8, particleSize: 0.25,
        lineAmount: 0.15, electronRate: 0.001,
        pulsePeriod: 3.0,
    },
    listening: {
        radius: 24, speed: 0.8, brightness: 1.0,
        color: 0x00ccff, particleSize: 0.30,
        lineAmount: 0.5, electronRate: 0.005,
        pulsePeriod: 0.8,
    },
    processing: {
        radius: 20, speed: 1.2, brightness: 1.2,
        color: 0x00d4aa, particleSize: 0.28,
        lineAmount: 0.7, electronRate: 0.012,
        pulsePeriod: 0.5,
    },
    speaking: {
        radius: 22, speed: 1.5, brightness: 1.4,
        color: 0x00ffcc, particleSize: 0.32,
        lineAmount: 0.85, electronRate: 0.006,
        pulsePeriod: 0.3,
    },
    disconnected: {
        radius: 18, speed: 0.1, brightness: 0.3,
        color: 0x993333, particleSize: 0.18,
        lineAmount: 0.0, electronRate: 0.0,
        pulsePeriod: 5.0,
    },
};

const N = 2000;               // particle count
const MAX_LINES = 6000;       // max connection lines (each line = 2 endpoints)
const MAX_ELECTRONS = 150;    // max travelling bright dots

export class DaisyOrb {
    constructor(canvas) {
        this.canvas = canvas;
        this._state = 'idle';
        this._target = { ...STATE_PARAMS['idle'] };
        this._current = { ...STATE_PARAMS['idle'] };
        this._lerpSpeed = 2.5;
        this._lastTime = performance.now();

        // Audio state
        this._audioAmp = 0;       // current amplitude (envelope or analyser)
        this._analyser = null;    // Web Audio AnalyserNode (Phase F8)
        this._freqData = null;    // Uint8Array for frequency data

        // Transition energy — spikes on state change for a "tumble" feel
        this._transitionEnergy = 0;

        // Envelope playback
        this._envelope = null;
        this._envelopeDuration = 0;
        this._envelopeElapsed = 0;

        // Per-particle state
        this._positions = null;   // Float32Array(N*3)
        this._velocities = null;  // Float32Array(N*3)
        this._phases = null;      // Float32Array(N) — random phase per particle

        // Electrons
        this._activeElectrons = [];
        this._activeConnections = []; // pool of [startIdx, endIdx] for electron spawning
        this._electronSpawnTimer = 0;

        this._initRenderer();
        this._initScene();
        this._initParticles();
        this._initLines();
        this._initElectrons();
        this._initPostProcessing();
        this._resize();
        window.addEventListener('resize', () => this._resize());

        this._animate = this._animate.bind(this);
        requestAnimationFrame(this._animate);
    }

    // ====================================================================
    // Init
    // ====================================================================

    _initRenderer() {
        this._renderer = new THREE.WebGLRenderer({
            canvas: this.canvas, alpha: false, antialias: true,
            powerPreference: 'high-performance',
        });
        this._renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this._renderer.setClearColor(0x050508);
    }

    _initScene() {
        this._scene = new THREE.Scene();
        this._camera = new THREE.PerspectiveCamera(45, 2, 0.1, 200);
        this._camera.position.z = 80;
        this._camera.lookAt(0, 0, 0);
    }

    _initParticles() {
        this._positions = new Float32Array(N * 3);
        this._velocities = new Float32Array(N * 3);
        this._phases = new Float32Array(N);

        const geom = new THREE.BufferGeometry();
        const colors = new Float32Array(N * 3);

        for (let i = 0; i < N; i++) {
            // Distribute within a sphere of radius 25
            const r = 25 * Math.cbrt(Math.random());
            const theta = Math.random() * Math.PI * 2;
            const phi = Math.acos(2 * Math.random() - 1);
            this._positions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
            this._positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
            this._positions[i * 3 + 2] = r * Math.cos(phi);

            this._velocities[i * 3]     = 0;
            this._velocities[i * 3 + 1] = 0;
            this._velocities[i * 3 + 2] = 0;

            this._phases[i] = Math.random() * Math.PI * 2;

            // Slightly varied base color
            colors[i * 3]     = 0.3 + Math.random() * 0.7;
            colors[i * 3 + 1] = 0.6 + Math.random() * 0.4;
            colors[i * 3 + 2] = 0.8 + Math.random() * 0.2;
        }

        geom.setAttribute('position', new THREE.BufferAttribute(this._positions, 3));
        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        this._particleMat = new THREE.PointsMaterial({
            size: 0.25,
            vertexColors: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            transparent: true,
            opacity: 0.8,
        });
        this._points = new THREE.Points(geom, this._particleMat);
        this._scene.add(this._points);
    }

    _initLines() {
        const linePositions = new Float32Array(MAX_LINES * 6); // 2 endpoints * 3 coords
        const lineGeom = new THREE.BufferGeometry();
        lineGeom.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
        lineGeom.setDrawRange(0, 0);
        this._linePositions = linePositions;

        const lineMat = new THREE.LineBasicMaterial({
            color: 0x4ca8e8,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
            transparent: true,
            opacity: 0,
        });
        this._lineMat = lineMat;
        this._lines = new THREE.LineSegments(lineGeom, lineMat);
        this._scene.add(this._lines);
    }

    _initElectrons() {
        const electronPositions = new Float32Array(MAX_ELECTRONS * 3);
        const electronGeom = new THREE.BufferGeometry();
        electronGeom.setAttribute('position', new THREE.BufferAttribute(electronPositions, 3));
        electronGeom.setDrawRange(0, 0);
        this._electronPositions = electronPositions;

        const electronMat = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.35,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });
        this._electrons = new THREE.Points(electronGeom, electronMat);
        this._scene.add(this._electrons);
    }

    _initPostProcessing() {
        this._composer = new EffectComposer(this._renderer);
        this._composer.addPass(new RenderPass(this._scene, this._camera));
        this._bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            0.6, 0.4, 0.25,
        );
        this._composer.addPass(this._bloomPass);
    }

    _resize() {
        const w = window.innerWidth;
        const h = window.innerHeight;
        this._renderer.setSize(w, h);
        this._composer.setSize(w, h);
        this._camera.aspect = w / Math.max(h, 1);
        this._camera.updateProjectionMatrix();
    }

    // ====================================================================
    // Public API
    // ====================================================================

    setState(stateName) {
        if (stateName === this._state) return;
        this._state = stateName;
        this._target = { ...(STATE_PARAMS[stateName] || STATE_PARAMS['idle']) };
        // Spike transition energy for visual tumble
        this._transitionEnergy = 1.0;
    }

    onSentence() {
        // Brief energy spike when a sentence arrives
        this._transitionEnergy = Math.min(1.0, this._transitionEnergy + 0.4);
    }

    onAudioEnvelope(envelope, durationS) {
        this._envelope = envelope;
        this._envelopeDuration = durationS;
        this._envelopeElapsed = 0;
    }

    /** Set a Web Audio AnalyserNode for real-time frequency data (Phase F8). */
    setAnalyser(analyser) {
        this._analyser = analyser;
        if (analyser) {
            this._freqData = new Uint8Array(analyser.frequencyBinCount);
        } else {
            this._freqData = null;
        }
    }

    dispose() {
        this._renderer.dispose();
        this._points.geometry.dispose();
        this._particleMat.dispose();
        this._lines.geometry.dispose();
        this._lineMat.dispose();
        this._electrons.geometry.dispose();
        if (this._electrons.material) this._electrons.material.dispose();
    }

    // ====================================================================
    // Render loop
    // ====================================================================

    _animate(timestamp) {
        requestAnimationFrame(this._animate);

        const dt = Math.min((timestamp - this._lastTime) / 1000, 0.1);
        this._lastTime = timestamp;

        this._updateAudio(dt);
        this._lerpParams(dt);
        this._updateParticles(dt);
        this._buildConnections();
        this._updateElectrons(dt);
        this._composer.render();
    }

    // --- Audio ---

    _updateAudio(dt) {
        // Read from AnalyserNode if available (Phase F8), else use envelope
        if (this._analyser && this._freqData) {
            this._analyser.getByteFrequencyData(this._freqData);
            // Bass: average of first 8 frequency bins, normalised to 0-1
            let bass = 0;
            const bins = Math.min(8, this._freqData.length);
            for (let i = 0; i < bins; i++) bass += this._freqData[i];
            bass /= bins * 255;
            // Mid: bins 8-24
            let mid = 0;
            const midEnd = Math.min(24, this._freqData.length);
            for (let i = bins; i < midEnd; i++) mid += this._freqData[i];
            mid /= (midEnd - bins) * 255;
            // Blend toward analyser values
            this._audioAmp += (bass - this._audioAmp) * Math.min(1, dt * 20);
            this._audioMid = (this._audioMid || 0) + (mid - (this._audioMid || 0)) * Math.min(1, dt * 20);
        } else if (this._envelope && this._envelopeElapsed < this._envelopeDuration) {
            // Play through pre-computed envelope
            this._envelopeElapsed += dt;
            const progress = Math.min(1, this._envelopeElapsed / this._envelopeDuration);
            const idx = Math.floor(progress * (this._envelope.length - 1));
            const frac = (progress * (this._envelope.length - 1)) - idx;
            const a = this._envelope[idx] || 0;
            const b = this._envelope[Math.min(idx + 1, this._envelope.length - 1)] || 0;
            const targetAmp = a + (b - a) * frac;
            this._audioAmp += (targetAmp - this._audioAmp) * Math.min(1, dt * 30);
        } else {
            // Decay to silence
            this._audioAmp += (0 - this._audioAmp) * Math.min(1, dt * 6);
            this._audioMid = (this._audioMid || 0) * (1 - Math.min(1, dt * 6));
        }
    }

    // --- Parameter lerp ---

    _lerpParams(dt) {
        const t = 1 - Math.exp(-this._lerpSpeed * dt);
        const c = this._current;
        const tg = this._target;
        for (const key of Object.keys(c)) {
            c[key] += (tg[key] - c[key]) * t;
        }

        // Decay transition energy
        this._transitionEnergy *= Math.pow(0.985, dt * 60);

        // Bloom tracks brightness
        this._bloomPass.strength = c.brightness * 0.9;
    }

    // --- Particles ---

    _updateParticles(dt) {
        const pos = this._positions;
        const vel = this._velocities;
        const ph = this._phases;
        const cr = this._current.radius;
        const cs = this._current.speed;
        const elapsed = timestamp => this._lastTime * 0.001; // approximate for sine

        const t = this._lastTime * 0.001;
        const bass = this._audioAmp;
        const mid = this._audioMid || 0;
        const isSpeaking = this._state === 'speaking';

        for (let i = 0; i < N; i++) {
            const i3 = i * 3;
            const px = pos[i3];
            const py = pos[i3 + 1];
            const pz = pos[i3 + 2];

            // Distance from centre
            const dist = Math.sqrt(px * px + py * py + pz * pz) || 0.001;
            const nx = px / dist;
            const ny = py / dist;
            const nz = pz / dist;

            // --- Velocity perturbation (organic wobble) ---
            vel[i3]     += Math.sin(t * 3.7 + ph[i]) * cs * 0.15 * dt;
            vel[i3 + 1] += Math.cos(t * 2.9 + ph[i]) * cs * 0.15 * dt;
            vel[i3 + 2] += Math.sin(t * 4.1 + ph[i]) * cs * 0.15 * dt;

            // --- Radial force: pull toward target radius ---
            const radialForce = (dist - cr) * 0.08;
            vel[i3]     -= nx * radialForce * dt;
            vel[i3 + 1] -= ny * radialForce * dt;
            vel[i3 + 2] -= nz * radialForce * dt;

            // --- Audio-driven radial push ---
            if (bass > 0.02) {
                vel[i3]     += nx * bass * 2.5 * dt;
                vel[i3 + 1] += ny * bass * 2.5 * dt;
                vel[i3 + 2] += nz * bass * 2.5 * dt;
            }

            // Speaking state: mid-frequency pulsing
            if (isSpeaking && mid > 0.05) {
                const pulse = Math.sin(t * 8 + ph[i]) * mid * 0.6;
                vel[i3]     += nx * pulse * dt;
                vel[i3 + 1] += ny * pulse * dt;
                vel[i3 + 2] += nz * pulse * dt;
            }

            // --- Damping ---
            vel[i3]     *= 0.992;
            vel[i3 + 1] *= 0.992;
            vel[i3 + 2] *= 0.992;

            // --- Update position ---
            pos[i3]     += vel[i3] * dt;
            pos[i3 + 1] += vel[i3 + 1] * dt;
            pos[i3 + 2] += vel[i3 + 2] * dt;
        }

        this._points.geometry.attributes.position.needsUpdate = true;

        // Visual tweaks
        this._particleMat.opacity = this._current.brightness * 0.6 + bass * 0.08;
        this._particleMat.size = this._current.particleSize + bass * 0.04;
    }

    // --- Connection lines ---

    _buildConnections() {
        const la = this._current.lineAmount;
        if (la < 0.01) {
            this._lines.geometry.setDrawRange(0, 0);
            this._lineMat.opacity = 0;
            this._activeConnections.length = 0;
            return;
        }

        const pos = this._positions;
        // Distance threshold scales with audio amplitude
        const maxDist = 8 + this._audioAmp * 15;
        const maxDistSq = maxDist * maxDist;
        const step = Math.max(1, Math.floor(N / 500));
        let lineCount = 0;
        this._activeConnections.length = 0;

        for (let i = 0; i < N; i += step) {
            for (let j = i + step; j < N; j += step) {
                const i3 = i * 3;
                const j3 = j * 3;
                const dx = pos[i3] - pos[j3];
                const dy = pos[i3 + 1] - pos[j3 + 1];
                const dz = pos[i3 + 2] - pos[j3 + 2];
                const distSq = dx * dx + dy * dy + dz * dz;

                if (distSq < maxDistSq && lineCount < MAX_LINES) {
                    const lc = lineCount * 6;
                    this._linePositions[lc]     = pos[i3];
                    this._linePositions[lc + 1] = pos[i3 + 1];
                    this._linePositions[lc + 2] = pos[i3 + 2];
                    this._linePositions[lc + 3] = pos[j3];
                    this._linePositions[lc + 4] = pos[j3 + 1];
                    this._linePositions[lc + 5] = pos[j3 + 2];
                    lineCount++;

                    // Store first 400 for electron spawning
                    if (this._activeConnections.length < 400) {
                        this._activeConnections.push([i3 / 3, j3 / 3]);
                    }
                }
            }
        }

        this._lines.geometry.attributes.position.needsUpdate = true;
        this._lines.geometry.setDrawRange(0, lineCount * 2);
        this._lineMat.opacity = la * 0.12;
    }

    // --- Electrons ---

    _updateElectrons(dt) {
        // Spawn new electrons
        const er = this._current.electronRate;
        this._electronSpawnTimer += dt;
        if (er > 0 && this._electronSpawnTimer > 1 / Math.max(er * 60, 1) && this._activeConnections.length > 0) {
            this._electronSpawnTimer = 0;
            if (this._activeElectrons.length < MAX_ELECTRONS) {
                const connIdx = Math.floor(Math.random() * this._activeConnections.length);
                const [startIdx, endIdx] = this._activeConnections[connIdx];
                const si = startIdx * 3;
                const ei = endIdx * 3;
                this._activeElectrons.push({
                    sx: this._positions[si], sy: this._positions[si + 1], sz: this._positions[si + 2],
                    ex: this._positions[ei], ey: this._positions[ei + 1], ez: this._positions[ei + 2],
                    t: 0,
                    speed: 0.3 + Math.random() * 0.6, // 0.3-0.9 progress per second
                });
            }
        }

        // Update and cull electrons
        let electronCount = 0;
        for (let i = this._activeElectrons.length - 1; i >= 0; i--) {
            const el = this._activeElectrons[i];
            el.t += el.speed * dt;
            if (el.t >= 1) {
                this._activeElectrons.splice(i, 1);
            } else {
                const ei = electronCount * 3;
                this._electronPositions[ei]     = el.sx + (el.ex - el.sx) * el.t;
                this._electronPositions[ei + 1] = el.sy + (el.ey - el.sy) * el.t;
                this._electronPositions[ei + 2] = el.sz + (el.ez - el.sz) * el.t;
                electronCount++;
            }
        }

        this._electrons.geometry.attributes.position.needsUpdate = true;
        this._electrons.geometry.setDrawRange(0, electronCount);
    }
}
