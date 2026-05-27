/* D.A.I.S.Y. v2 — Application
 *
 * Wires the Three.js orb, WebSocket client, and UI together.
 * All communication with the backend happens over a single WebSocket.
 */

import { DaisyOrb } from './orb.js';

// --- DOM references ---
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const voiceBtn = document.getElementById('voice-btn');
const statusLabel = document.getElementById('status-label');
const statusDot = document.getElementById('status-dot');
const transcriptContainer = document.getElementById('transcript-container');
const connectingOverlay = document.getElementById('connecting-overlay');
const historyContent = document.getElementById('history-content');
const memoryContent = document.getElementById('memory-content');
const settingsContent = document.getElementById('settings-content');
const chatPanel = document.getElementById('chat-panel');
const chatToggle = document.getElementById('chat-toggle');
const menuBtn = document.getElementById('menu-btn');
const menuDropdown = document.getElementById('menu-dropdown');

// --- State ---
let currentState = 'idle';
let ws = null;
let reconnectTimer = null;
let orb = null;

// --- Orb ---
const orbCanvas = document.getElementById('orb-canvas');
if (orbCanvas) {
    orb = new DaisyOrb(orbCanvas);
}

// --- Chat toggle ---
function toggleChat() {
    if (!chatPanel) return;
    const isCollapsed = chatPanel.classList.contains('collapsed');
    if (isCollapsed) {
        chatPanel.classList.remove('collapsed');
        if (chatToggle) chatToggle.classList.add('moved-up');
        textInput.focus();
    } else {
        chatPanel.classList.add('collapsed');
        if (chatToggle) chatToggle.classList.remove('moved-up');
        textInput.blur();
    }
}

if (chatToggle) {
    chatToggle.addEventListener('click', toggleChat);
}

// Close chat on Escape when input is focused
document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && document.activeElement === textInput) {
        if (chatPanel && !chatPanel.classList.contains('collapsed')) {
            toggleChat();
        }
    }
});

// --- Hamburger menu ---
function toggleMenu() {
    if (!menuDropdown) return;
    menuDropdown.classList.toggle('hidden');
}

function closeMenu() {
    if (menuDropdown) menuDropdown.classList.add('hidden');
}

if (menuBtn) {
    menuBtn.addEventListener('click', function (e) {
        e.stopPropagation();
        toggleMenu();
    });
}

// Close menu on outside click
document.addEventListener('click', function (e) {
    if (menuDropdown && !menuDropdown.classList.contains('hidden')) {
        if (!menuBtn.contains(e.target) && !menuDropdown.contains(e.target)) {
            closeMenu();
        }
    }
});

// Close menu on Escape
document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeMenu();
});

// Menu items open panels and close the menu — wired by the global
// [data-panel] query above; we just need to also close the menu.
// We wrap openPanel to intercept clicks from inside the dropdown.
if (menuDropdown) {
    menuDropdown.addEventListener('click', function (e) {
        // Let the [data-panel] handler on the button run; then close menu
        if (e.target.closest('[data-panel]')) {
            // small delay so the panel opens before we potentially steal focus
            setTimeout(closeMenu, 50);
        }
    });
}

// Auto-expand chat when user sends a message
function ensureChatOpen() {
    if (chatPanel && chatPanel.classList.contains('collapsed')) {
        chatPanel.classList.remove('collapsed');
    }
}

// --- Panel toggle ---
function openPanel(id) {
    const panel = document.getElementById(id);
    if (!panel) return;
    const visible = !panel.classList.contains('hidden');

    document.querySelectorAll('.slide-panel').forEach(p => p.classList.add('hidden'));

    if (!visible) {
        panel.classList.remove('hidden');
        if (id === 'panel-history') loadHistory();
        else if (id === 'panel-memory') loadMemory();
        else if (id === 'panel-settings') loadSettings();
    }
}

function closeAllPanels() {
    document.querySelectorAll('.slide-panel').forEach(p => p.classList.add('hidden'));
}

document.querySelectorAll('[data-panel]').forEach(btn => {
    btn.addEventListener('click', () => openPanel(btn.dataset.panel));
});

document.querySelectorAll('.panel-close').forEach(btn => {
    btn.addEventListener('click', () => {
        btn.closest('.slide-panel').classList.add('hidden');
    });
});

document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeAllPanels();
});

// --- Status display ---
function setStatus(state) {
    if (currentState === state) return;
    currentState = state;
    statusLabel.textContent = state.toUpperCase();

    // Update status dot color via CSS custom property
    const colors = {
        idle: '#0a84ff',
        listening: '#00ccff',
        processing: '#00d4aa',
        speaking: '#00ffcc',
        disconnected: '#cc3333',
    };
    statusDot.style.background = colors[state] || colors.idle;
    statusDot.style.boxShadow = `0 0 8px ${colors[state] || colors.idle}`;
}

// --- Transcript ---
function appendMessage(role, text) {
    const el = document.createElement('div');
    el.className = 'message message-' + role;
    el.textContent = text;
    transcriptContainer.appendChild(el);
    const area = transcriptContainer.parentElement;
    if (area) area.scrollTop = area.scrollHeight;
}

// --- Send text ---
function sendText() {
    const text = textInput.value.trim();
    if (!text) return;

    ensureChatOpen();
    appendMessage('user', text);
    textInput.value = '';

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'text_input', text }));
    }
}

sendBtn.addEventListener('click', sendText);
textInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') sendText();
});

// --- Voice toggle (stub — Phase F8) ---
voiceBtn.addEventListener('click', function () {
    this.classList.toggle('active');
});

// --- Panel content loaders ---
function loadHistory() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'get_history' }));
    } else {
        historyContent.innerHTML = '<p style="color: var(--text-dim);">Not connected.</p>';
    }
}

function loadMemory() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'get_memory' }));
    } else {
        memoryContent.innerHTML = '<p style="color: var(--text-dim);">Not connected.</p>';
    }
}

function loadSettings() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'get_config' }));
    } else {
        settingsContent.innerHTML = '<p style="color: var(--text-dim);">Not connected.</p>';
    }
}

// --- WebSocket ---
function connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = protocol + '//' + location.host + '/ws';

    ws = new WebSocket(wsUrl);

    ws.onopen = function () {
        if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        if (connectingOverlay) connectingOverlay.classList.add('fade-out');
        ws.send(JSON.stringify({ type: 'get_config' }));
    };

    ws.onmessage = function (event) {
        const msg = JSON.parse(event.data);

        switch (msg.type) {
            case 'state':
                setStatus(msg.state);
                if (orb) orb.setState(msg.state);
                break;
            case 'sentence':
                appendMessage('daisy', msg.text);
                if (orb) orb.onSentence();
                break;
            case 'transcript':
                // Full transcript display uses msg.text (final/partial fields)
                break;
            case 'response_complete':
                break;
            case 'audio_amplitude':
                if (orb && msg.envelope && msg.duration_s) {
                    orb.onAudioEnvelope(msg.envelope, msg.duration_s);
                }
                break;
            case 'history':
                renderHistory(msg.turns);
                break;
            case 'memory':
                renderMemory(msg.facts);
                break;
            case 'config':
                renderSettings(msg.settings);
                break;
            case 'text_accepted':
                // Message was received by the server
                break;
            case 'error':
                appendMessage('daisy', '⚠ ' + msg.message);
                break;
        }
    };

    ws.onclose = function () {
        setStatus('disconnected');
        if (orb) orb.setState('disconnected');
        if (connectingOverlay) connectingOverlay.classList.remove('fade-out');
        const delay = reconnectTimer ? 2000 : 1000;
        reconnectTimer = setTimeout(connect, delay);
    };

    ws.onerror = function () {
        // onclose fires after onerror; reconnect handled there
    };
}

// --- Panel renderers ---

let _allFacts = []; // cached for client-side search

function renderHistory(turns) {
    if (!turns || turns.length === 0) {
        historyContent.innerHTML = '<p style="color: var(--text-dim);">No history yet.</p>';
        return;
    }
    // Show newest first
    const items = turns.slice().reverse();
    historyContent.innerHTML = items.map(t => {
        const roleLabel = t.role === 'assistant' ? 'DAISY' : 'Boss';
        const roleClass = t.role === 'assistant' ? 'role-daisy' : '';
        return (
            '<div class="turn-item">' +
            '<div class="turn-role ' + roleClass + '">' + roleLabel + '</div>' +
            '<div class="turn-content">' + (t.content || '') + '</div>' +
            '</div>'
        );
    }).join('');
}

function renderMemory(facts) {
    _allFacts = facts || [];
    _renderMemoryList(_allFacts);
}

function _renderMemoryList(facts) {
    if (!facts || facts.length === 0) {
        memoryContent.innerHTML =
            '<input class="search-input" id="memory-search" placeholder="Search facts..." style="display:none">' +
            '<p style="color: var(--text-dim);">No facts stored yet.</p>';
        return;
    }
    const listHtml = facts.map(f =>
        '<div class="fact-item">' +
        '<div class="fact-body">' +
        '<div class="fact-key">' + (f.key || '') + '</div>' +
        '<div class="fact-value">' + (f.value || '') + '</div>' +
        (f.category && f.category !== 'general'
            ? '<span class="fact-category">' + f.category + '</span>'
            : '') +
        '</div>' +
        '<button class="delete-btn" data-fact-key="' + (f.key || '') + '" title="Delete">×</button>' +
        '</div>'
    ).join('');

    memoryContent.innerHTML =
        '<input class="search-input" id="memory-search" placeholder="Search facts...">' +
        '<div id="memory-list">' + listHtml + '</div>';

    // Wire search
    const searchInput = document.getElementById('memory-search');
    if (searchInput) {
        searchInput.addEventListener('input', function () {
            const q = this.value.toLowerCase().trim();
            if (!q) {
                _renderMemoryList(_allFacts);
                return;
            }
            const filtered = _allFacts.filter(f =>
                (f.key || '').toLowerCase().includes(q) ||
                (f.value || '').toLowerCase().includes(q)
            );
            _renderMemoryList(filtered);
        });
    }

    // Wire delete buttons
    document.querySelectorAll('#memory-list .delete-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            const key = this.dataset.factKey;
            if (ws && ws.readyState === WebSocket.OPEN) {
                fetch('/api/memory/' + encodeURIComponent(key), { method: 'DELETE' })
                    .then(r => { if (r.ok) loadMemory(); })
                    .catch(() => {});
            }
        });
    });
}

function renderSettings(settings) {
    if (!settings) {
        settingsContent.innerHTML = '<p style="color: var(--text-dim);">No settings available.</p>';
        return;
    }
    const mode = settings.mode || 'wake_word';
    const vadThreshold = (settings.vad && settings.vad.silero_threshold) || 0.5;
    const wakeThreshold = (settings.wake_word && settings.wake_word.threshold) || 0.4;
    const ttsVoice = (settings.tts && settings.tts.voice) || 'af_heart';
    const toolsEnabled = (settings.tools && settings.tools.enabled) ? 'Enabled' : 'Disabled';

    settingsContent.innerHTML =
        '<div class="setting-row">' +
        '  <div><label>Mode</label><div class="hint">How DAISY activates</div></div>' +
        '  <select id="cfg-mode">' +
        '    <option value="wake_word"' + (mode === 'wake_word' ? ' selected' : '') + '>Wake Word</option>' +
        '    <option value="always_on"' + (mode === 'always_on' ? ' selected' : '') + '>Always On</option>' +
        '    <option value="push_to_talk"' + (mode === 'push_to_talk' ? ' selected' : '') + '>Push to Talk</option>' +
        '  </select>' +
        '</div>' +
        '<div class="setting-row">' +
        '  <div><label>VAD Sensitivity</label><div class="hint">Speech detection threshold</div></div>' +
        '  <input type="range" id="cfg-vad" min="0.1" max="0.9" step="0.05" value="' + vadThreshold + '">' +
        '  <span class="setting-value" id="cfg-vad-val">' + vadThreshold.toFixed(2) + '</span>' +
        '</div>' +
        '<div class="setting-row">' +
        '  <div><label>Wake Threshold</label><div class="hint">Wake word detection confidence</div></div>' +
        '  <input type="range" id="cfg-wake" min="0.1" max="0.9" step="0.05" value="' + wakeThreshold + '">' +
        '  <span class="setting-value" id="cfg-wake-val">' + wakeThreshold.toFixed(2) + '</span>' +
        '</div>' +
        '<div class="setting-row">' +
        '  <div><label>TTS Voice</label><div class="hint">Kokoro voice preset</div></div>' +
        '  <select id="cfg-voice">' +
        '    <option value="af_heart"' + (ttsVoice === 'af_heart' ? ' selected' : '') + '>af_heart (F)</option>' +
        '    <option value="af_bella"' + (ttsVoice === 'af_bella' ? ' selected' : '') + '>af_bella (F)</option>' +
        '    <option value="af_nicole"' + (ttsVoice === 'af_nicole' ? ' selected' : '') + '>af_nicole (F)</option>' +
        '    <option value="af_sarah"' + (ttsVoice === 'af_sarah' ? ' selected' : '') + '>af_sarah (F)</option>' +
        '    <option value="af_sky"' + (ttsVoice === 'af_sky' ? ' selected' : '') + '>af_sky (F)</option>' +
        '    <option value="am_adam"' + (ttsVoice === 'am_adam' ? ' selected' : '') + '>am_adam (M)</option>' +
        '    <option value="am_michael"' + (ttsVoice === 'am_michael' ? ' selected' : '') + '>am_michael (M)</option>' +
        '  </select>' +
        '</div>' +
        '<div class="setting-row">' +
        '  <div><label>Tools</label><div class="hint">Web search, shell, files</div></div>' +
        '  <span style="color: var(--text); font-size: 13px;">' + toolsEnabled + '</span>' +
        '</div>' +
        '<button class="save-btn" id="cfg-save">Save Settings</button>';

    // Wire sliders
    const vadSlider = document.getElementById('cfg-vad');
    const vadVal = document.getElementById('cfg-vad-val');
    if (vadSlider && vadVal) {
        vadSlider.addEventListener('input', () => { vadVal.textContent = parseFloat(vadSlider.value).toFixed(2); });
    }
    const wakeSlider = document.getElementById('cfg-wake');
    const wakeVal = document.getElementById('cfg-wake-val');
    if (wakeSlider && wakeVal) {
        wakeSlider.addEventListener('input', () => { wakeVal.textContent = parseFloat(wakeSlider.value).toFixed(2); });
    }

    // Wire save
    const saveBtn = document.getElementById('cfg-save');
    if (saveBtn) {
        saveBtn.addEventListener('click', function () {
            const mode = document.getElementById('cfg-mode').value;
            const vadThreshold = parseFloat(document.getElementById('cfg-vad').value);
            const wakeThreshold = parseFloat(document.getElementById('cfg-wake').value);
            const voice = document.getElementById('cfg-voice').value;

            fetch('/api/config', {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: mode,
                    vad: { silero_threshold: vadThreshold },
                    wake_word: { threshold: wakeThreshold },
                    tts: { voice: voice },
                }),
            }).then(r => r.json()).then(d => {
                saveBtn.textContent = d.changed ? 'Saved!' : 'No changes';
                saveBtn.style.opacity = '0.7';
                setTimeout(() => {
                    saveBtn.textContent = 'Save Settings';
                    saveBtn.style.opacity = '1';
                }, 2000);
            }).catch(() => {
                saveBtn.textContent = 'Save failed';
                setTimeout(() => { saveBtn.textContent = 'Save Settings'; }, 2000);
            });
        });
    }
}

// --- Init ---
connect();
