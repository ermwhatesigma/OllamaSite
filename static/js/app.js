/**
 * OllamaGate — Frontend Application
 * Handles: model selection, chat CRUD, streaming, file uploads, PWA.
 */

"use strict";

// ─────────────────────────────────────────────
//  State
// ─────────────────────────────────────────────
const state = {
  activeChatId:   null,
  currentModel:   null,
  visionEnabled:  false,
  streaming:      false,
  attachedFile:   null,   // File object
  chats:          [],     // [{id, title, model, updated_at}]
};

// ─────────────────────────────────────────────
//  DOM refs
// ─────────────────────────────────────────────
const $ = id => document.getElementById(id);
const sidebar      = $('sidebar');
const overlay      = $('overlay');
const chatList     = $('chatList');
const newChatBtn   = $('newChatBtn');
const modelSelect  = $('modelSelect');
const visionTag    = $('visionTag');
const topbarTitle  = $('topbarTitle');
const messages     = $('messages');
const emptyState   = $('emptyState');
const msgWrap      = $('msgWrap');
const promptTA     = $('prompt');
const sendBtn      = $('sendBtn');
const attachBtn    = $('attachBtn');
const fileInput    = $('fileInput');
const filePreview  = $('filePreview');
const fileNameSpan = $('fileName');
const fileIconSpan = $('fileIcon');
const removeFile   = $('removeFile');
const menuBtn      = $('menuBtn');
const logoutBtn    = $('logoutBtn');
const timerBadge   = $('timerBadge');

// ─────────────────────────────────────────────
//  Sidebar toggle (mobile)
// ─────────────────────────────────────────────
menuBtn.addEventListener('click', () => {
  sidebar.classList.toggle('open');
  overlay.classList.toggle('show');
});
overlay.addEventListener('click', closeSidebar);
function closeSidebar() {
  sidebar.classList.remove('open');
  overlay.classList.remove('show');
}

// ─────────────────────────────────────────────
//  Logout
// ─────────────────────────────────────────────
logoutBtn.addEventListener('click', async () => {
  await fetch('/api/logout', { method: 'POST' });
  location.href = '/';
});

// ─────────────────────────────────────────────
//  Session timer
// ─────────────────────────────────────────────
async function refreshTimer() {
  try {
    const d = await (await fetch('/api/status')).json();
    if (!d.token_valid) {
      timerBadge.textContent = '⏰ Session expired';
      timerBadge.style.borderColor = 'rgba(255,78,106,.3)';
      timerBadge.style.color = 'var(--danger)';
      alert('Your session has expired. You will be redirected to the login page.');
      location.href = '/';
      return;
    }
    const m = Math.floor(d.expires_in / 60);
    const s = d.expires_in % 60;
    timerBadge.textContent = `⏰ ${m}m ${String(s).padStart(2,'0')}s remaining`;
  } catch { /* ignore */ }
}
setInterval(refreshTimer, 5000);
refreshTimer();

// ─────────────────────────────────────────────
//  Models
// ─────────────────────────────────────────────
async function loadModels() {
  try {
    const data = await (await fetch('/api/models')).json();
    const models = data.models || [];
    modelSelect.innerHTML = models.length
      ? models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('')
      : '<option value="">No models found</option>';

    if (models.length) {
      state.currentModel = models[0];
      updateVisionUI();
    }
  } catch {
    modelSelect.innerHTML = '<option value="">Ollama unreachable</option>';
  }
}

modelSelect.addEventListener('change', () => {
  state.currentModel = modelSelect.value;
  updateVisionUI();
});

async function updateVisionUI() {
  if (!state.currentModel) { state.visionEnabled = false; updateAttachBtn(); return; }
  try {
    const d = await (await fetch(`/api/model/vision/${encodeURIComponent(state.currentModel)}`)).json();
    state.visionEnabled = d.supports_vision;
  } catch {
    state.visionEnabled = false;
  }
  visionTag.style.display = state.visionEnabled ? 'inline-block' : 'none';
  updateAttachBtn();
}

function updateAttachBtn() {
  // Allow attaching only if vision model OR non-image file
  // We still allow PDFs/text always; image only for vision models
  attachBtn.title = state.visionEnabled
    ? 'Attach image, PDF, or text'
    : 'Attach PDF or text file (image upload disabled for this model)';
}

// ─────────────────────────────────────────────
//  Chat List
// ─────────────────────────────────────────────
async function loadChats() {
  const data = await (await fetch('/api/chats')).json();
  state.chats = data.chats || [];
  renderChatList();
}

function renderChatList() {
  chatList.innerHTML = state.chats.length
    ? state.chats.map(c => `
        <div class="chat-item ${c.id === state.activeChatId ? 'active' : ''}"
             data-id="${c.id}">
          <span>💬</span>
          <span class="chat-title" title="${esc(c.title)}">${esc(c.title)}</span>
          <button class="del-btn" data-id="${c.id}" title="Delete">🗑</button>
        </div>`).join('')
    : '<p style="padding:12px;font-size:12px;color:var(--muted);text-align:center">No chats yet</p>';

  chatList.querySelectorAll('.chat-item').forEach(el => {
    el.addEventListener('click', e => {
      if (e.target.classList.contains('del-btn')) return;
      loadChat(el.dataset.id);
      closeSidebar();
    });
  });
  chatList.querySelectorAll('.del-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      e.stopPropagation();
      deleteChat(btn.dataset.id);
    });
  });
}

async function deleteChat(id) {
  if (!confirm('Delete this chat?')) return;
  await fetch(`/api/chats/${id}`, { method: 'DELETE' });
  if (state.activeChatId === id) {
    state.activeChatId = null;
    msgWrap.innerHTML = '';
    showEmpty(true);
    topbarTitle.textContent = 'Select or start a chat';
  }
  await loadChats();
}

newChatBtn.addEventListener('click', async () => {
  if (!state.currentModel) { alert('Please wait for models to load.'); return; }
  const data = await (await fetch('/api/chats', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: state.currentModel, title: 'New Chat' }),
  })).json();
  state.activeChatId = data.id;
  msgWrap.innerHTML = '';
  showEmpty(false);
  topbarTitle.textContent = 'New Chat';
  await loadChats();
  closeSidebar();
  promptTA.focus();
});

// ─────────────────────────────────────────────
//  Load & render chat
// ─────────────────────────────────────────────
async function loadChat(id) {
  const data = await (await fetch(`/api/chats/${id}`)).json();
  state.activeChatId = id;
  state.currentModel  = data.chat.model;
  // Sync model select
  if ([...modelSelect.options].some(o => o.value === data.chat.model)) {
    modelSelect.value = data.chat.model;
    updateVisionUI();
  }
  topbarTitle.textContent = data.chat.title;
  msgWrap.innerHTML = '';
  showEmpty(false);
  data.messages.forEach(m => appendMessage(m.role, m.content, false));
  scrollToBottom();
  renderChatList();
  promptTA.focus();
}

// ─────────────────────────────────────────────
//  Message rendering
// ─────────────────────────────────────────────
function showEmpty(show) {
  emptyState.style.display  = show ? 'flex'  : 'none';
  msgWrap.style.display     = show ? 'none'  : 'block';
}

function appendMessage(role, content, animate = true) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.dataset.role = role;

  if (role === 'user') {
    div.innerHTML = `<div class="bubble">${esc(content)}</div>`;
  } else {
    div.innerHTML = `
      <div class="msg-meta">
        <div class="ai-avatar">🤖</div>
        <span>${esc(state.currentModel || 'AI')}</span>
      </div>
      <div class="bubble">
        <div class="bubble-md">${renderMarkdown(content)}</div>
      </div>`;
  }

  if (!animate) div.style.animation = 'none';
  msgWrap.appendChild(div);
  highlightCode(div);
  addCopyButtons(div);
  scrollToBottom();
  return div;
}

function createStreamingMessage() {
  showEmpty(false);
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = `
    <div class="msg-meta">
      <div class="ai-avatar">🤖</div>
      <span>${esc(state.currentModel || 'AI')}</span>
    </div>
    <div class="bubble">
      <div class="bubble-md typing-cursor"></div>
    </div>`;
  msgWrap.appendChild(div);
  scrollToBottom();
  return div.querySelector('.bubble-md');
}

function renderMarkdown(text) {
  if (typeof marked === 'undefined') return esc(text);
  marked.setOptions({ breaks: true, gfm: true });
  return marked.parse(text);
}

function highlightCode(el) {
  if (typeof hljs !== 'undefined') {
    el.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
  }
}

function addCopyButtons(el) {
  el.querySelectorAll('pre').forEach(pre => {
    if (pre.querySelector('.copy-code-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'copy-code-btn';
    btn.textContent = 'Copy';
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code');
      navigator.clipboard.writeText(code ? code.textContent : pre.textContent);
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
    });
    pre.style.position = 'relative';
    pre.appendChild(btn);
  });
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

// ─────────────────────────────────────────────
//  File attachment
// ─────────────────────────────────────────────
attachBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;

  // Client-side image check (server enforces too)
  if (file.type.startsWith('image/') && !state.visionEnabled) {
    alert('The selected model does not support images. Please switch to a vision-capable model first, or attach a PDF/text file.');
    fileInput.value = '';
    return;
  }

  const MAX = 10 * 1024 * 1024;
  if (file.size > MAX) {
    alert('File is too large. Maximum size is 10MB.');
    fileInput.value = '';
    return;
  }

  state.attachedFile = file;
  fileNameSpan.textContent = file.name;

  if (file.type.startsWith('image/')) {
    fileIconSpan.textContent = '🖼';
    // Show thumbnail
    const reader = new FileReader();
    reader.onload = e => {
      fileIconSpan.innerHTML = `<img src="${e.target.result}" style="max-height:40px;border-radius:5px;">`;
    };
    reader.readAsDataURL(file);
  } else if (file.type === 'application/pdf') {
    fileIconSpan.textContent = '📄';
  } else {
    fileIconSpan.textContent = '📝';
  }

  filePreview.classList.add('visible');
  fileInput.value = '';
});

removeFile.addEventListener('click', () => {
  state.attachedFile = null;
  filePreview.classList.remove('visible');
});

// ─────────────────────────────────────────────
//  Send message
// ─────────────────────────────────────────────
async function sendMessage() {
  if (state.streaming) return;
  const content = promptTA.value.trim();
  if (!content && !state.attachedFile) return;
  if (!state.activeChatId) {
    alert('Please create or select a chat first.');
    return;
  }
  if (!state.currentModel) {
    alert('Please select a model.');
    return;
  }

  state.streaming = true;
  sendBtn.disabled = true;
  promptTA.value = '';
  autoResizeTA();

  // Show user message immediately
  showEmpty(false);
  appendMessage('user', content);

  // Hide file preview
  const fileToSend = state.attachedFile;
  state.attachedFile = null;
  filePreview.classList.remove('visible');

  // Build request
  let fetchInit;
  const url = `/api/chats/${state.activeChatId}/message`;

  if (fileToSend) {
    const fd = new FormData();
    fd.append('content', content);
    fd.append('model',   state.currentModel);
    fd.append('file',    fileToSend);
    fetchInit = { method: 'POST', body: fd };
  } else {
    fetchInit = {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ content, model: state.currentModel }),
    };
  }

  // Create streaming bubble
  const streamEl = createStreamingMessage();
  let accumulated = '';

  try {
    const resp = await fetch(url, fetchInit);
    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      const errMsg  = errData.error || `Server error ${resp.status}`;
      streamEl.classList.remove('typing-cursor');
      streamEl.textContent = `⚠ ${errMsg}`;
      return;
    }

    const reader  = resp.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;
        let chunk;
        try { chunk = JSON.parse(raw); } catch { continue; }

        if (chunk.error) {
          streamEl.classList.remove('typing-cursor');
          streamEl.textContent = `⚠ ${chunk.error}`;
          return;
        }
        if (chunk.delta) {
          accumulated += chunk.delta;
          streamEl.innerHTML = renderMarkdown(accumulated);
          highlightCode(streamEl);
          addCopyButtons(streamEl.closest('.bubble'));
          scrollToBottom();
        }
        if (chunk.done) {
          streamEl.classList.remove('typing-cursor');
          // Reload chat list to update title
          loadChats();
        }
      }
    }
  } catch (err) {
    streamEl.classList.remove('typing-cursor');
    streamEl.textContent = `⚠ Connection error: ${err.message}`;
  } finally {
    state.streaming  = false;
    sendBtn.disabled = false;
    promptTA.focus();
  }
}

sendBtn.addEventListener('click', sendMessage);

promptTA.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// ─────────────────────────────────────────────
//  Auto-resize textarea
// ─────────────────────────────────────────────
function autoResizeTA() {
  promptTA.style.height = 'auto';
  promptTA.style.height = Math.min(promptTA.scrollHeight, 180) + 'px';
}
promptTA.addEventListener('input', autoResizeTA);

// ─────────────────────────────────────────────
//  Utility
// ─────────────────────────────────────────────
function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─────────────────────────────────────────────
//  Service Worker registration
// ─────────────────────────────────────────────
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js')
    .then(reg => console.log('[SW] Registered:', reg.scope))
    .catch(err => console.warn('[SW] Registration failed:', err));
}

// ─────────────────────────────────────────────
//  Init
// ─────────────────────────────────────────────
(async function init() {
  showEmpty(true);
  await loadModels();
  await loadChats();
})();
