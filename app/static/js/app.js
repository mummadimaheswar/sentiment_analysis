/* ===================================================================
   app.js — Multimodal Sentiment Intelligence Chat UI
   =================================================================== */

const API = "/api";

// ── State ──────────────────────────────────────────────────────────
let conversations = [];          // {id, title, messages[]}
let activeConvId = null;
let pendingAudio = null;         // File object
let pendingImage = null;         // File object
let isProcessing = false;

// ── DOM refs ───────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const messagesContainer = $(".messages-wrapper");
const chatContainer = $(".chat-container");
const messageInput = $("#messageInput");
const sendBtn = $("#sendBtn");
const filePreviewBar = $(".file-preview-bar");
const convList = $(".sidebar-conversations");

// ── Initialise ─────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  showWelcome();
  bindEvents();
});

function bindEvents() {
  sendBtn.addEventListener("click", handleSend);
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleSend(); }
  });
  messageInput.addEventListener("input", autoResize);

  $("#audioBtn").addEventListener("click", () => $("#audioUpload").click());
  $("#imageBtn").addEventListener("click", () => $("#imageUpload").click());
  $("#audioUpload").addEventListener("change", (e) => handleFileSelect(e, "audio"));
  $("#imageUpload").addEventListener("change", (e) => handleFileSelect(e, "image"));

  $(".new-chat-btn").addEventListener("click", newConversation);

  // Quick action cards
  $$(".quick-action").forEach((el) => {
    el.addEventListener("click", () => {
      messageInput.value = el.dataset.prompt;
      autoResize();
      messageInput.focus();
    });
  });
}

// ── Auto-resize textarea ───────────────────────────────────────────
function autoResize() {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + "px";
}

// ── File handling ──────────────────────────────────────────────────
function handleFileSelect(e, type) {
  const file = e.target.files[0];
  if (!file) return;
  if (type === "audio") pendingAudio = file;
  else pendingImage = file;
  updateFilePreview();
  e.target.value = "";
}

function removeFile(type) {
  if (type === "audio") pendingAudio = null;
  else pendingImage = null;
  updateFilePreview();
}

// Expose globally for inline onclick
window.removeFile = removeFile;

function updateFilePreview() {
  const chips = [];
  if (pendingAudio) {
    chips.push(`<span class="file-preview">🎙️ ${pendingAudio.name}
      <span class="remove-file" onclick="removeFile('audio')">✕</span></span>`);
  }
  if (pendingImage) {
    chips.push(`<span class="file-preview">🖼️ ${pendingImage.name}
      <span class="remove-file" onclick="removeFile('image')">✕</span></span>`);
  }
  filePreviewBar.innerHTML = chips.join("");
  filePreviewBar.classList.toggle("has-files", chips.length > 0);
}

// ── Conversations ──────────────────────────────────────────────────
function newConversation() {
  const conv = {
    id: Date.now().toString(),
    title: "New analysis",
    messages: [],
  };
  conversations.unshift(conv);
  activeConvId = conv.id;
  renderConversationList();
  showWelcome();
  messageInput.value = "";
  pendingAudio = null;
  pendingImage = null;
  updateFilePreview();
}

function renderConversationList() {
  if (conversations.length === 0) {
    convList.innerHTML = `<div class="conv-section-title">No conversations yet</div>`;
    return;
  }
  convList.innerHTML = `<div class="conv-section-title">Recent</div>` +
    conversations.map((c) => `
      <div class="conv-item ${c.id === activeConvId ? "active" : ""}"
           onclick="switchConversation('${c.id}')">
        ${escapeHtml(c.title)}
      </div>`).join("");
}

window.switchConversation = function (id) {
  activeConvId = id;
  renderConversationList();
  renderMessages();
};

// ── Render ─────────────────────────────────────────────────────────
function showWelcome() {
  messagesContainer.innerHTML = `
    <div class="welcome-screen">
      <div class="welcome-icon">🧠</div>
      <h2>Sentiment Intelligence</h2>
      <p>Analyse customer feedback across text, audio, and images.
         I combine RoBERTa, Wav2Vec2, and CLIP to detect emotions,
         sentiment, and issues in real time.</p>
      <div class="quick-actions">
        <div class="quick-action" data-prompt="I've been waiting 3 weeks for my order and it still hasn't arrived. This is unacceptable!">
          <div class="qa-icon">📦</div>
          <div class="qa-title">Delivery complaint</div>
          <div class="qa-desc">Analyse a frustrated shipping review</div>
        </div>
        <div class="quick-action" data-prompt="The product quality is amazing, I love everything about it! Fast delivery and perfect packaging.">
          <div class="qa-icon">⭐</div>
          <div class="qa-title">Positive review</div>
          <div class="qa-desc">Analyse a happy customer review</div>
        </div>
        <div class="quick-action" data-prompt="The app keeps crashing every time I try to make a payment. Very frustrated and considering switching.">
          <div class="qa-icon">🐛</div>
          <div class="qa-title">App bug report</div>
          <div class="qa-desc">Analyse a technical complaint</div>
        </div>
        <div class="quick-action" data-prompt="I was overcharged $50 on my last invoice and nobody in support can help me. This is ridiculous!">
          <div class="qa-icon">💳</div>
          <div class="qa-title">Billing issue</div>
          <div class="qa-desc">Analyse a billing complaint</div>
        </div>
      </div>
    </div>`;

  // Re-bind quick actions
  $$(".quick-action").forEach((el) => {
    el.addEventListener("click", () => {
      messageInput.value = el.dataset.prompt;
      autoResize();
      messageInput.focus();
    });
  });
}

function renderMessages() {
  const conv = conversations.find((c) => c.id === activeConvId);
  if (!conv || conv.messages.length === 0) { showWelcome(); return; }

  messagesContainer.innerHTML = conv.messages.map(renderMessage).join("");
  scrollToBottom();
}

function renderMessage(msg) {
  if (msg.role === "user") return renderUserMessage(msg);
  return renderAssistantMessage(msg);
}

function renderUserMessage(msg) {
  const attachments = [];
  if (msg.audioName) attachments.push(`<span class="attachment-chip"><span class="chip-icon">🎙️</span>${escapeHtml(msg.audioName)}</span>`);
  if (msg.imageName) attachments.push(`<span class="attachment-chip"><span class="chip-icon">🖼️</span>${escapeHtml(msg.imageName)}</span>`);

  return `
    <div class="message user">
      <div class="message-avatar">U</div>
      <div class="message-body">
        <div class="message-header">
          <span class="message-name">You</span>
          <span class="message-time">${msg.time}</span>
        </div>
        <div class="message-content"><p>${escapeHtml(msg.content)}</p></div>
        ${attachments.length ? `<div class="message-attachments">${attachments.join("")}</div>` : ""}
      </div>
    </div>`;
}

function renderAssistantMessage(msg) {
  let body = `<p>${escapeHtml(msg.content)}</p>`;
  if (msg.analysis) body += renderAnalysisCard(msg.analysis);
  return `
    <div class="message assistant">
      <div class="message-avatar">AI</div>
      <div class="message-body">
        <div class="message-header">
          <span class="message-name">Sentiment AI</span>
          <span class="message-time">${msg.time}</span>
        </div>
        <div class="message-content">${body}</div>
      </div>
    </div>`;
}

function renderAnalysisCard(a) {
  const sentClass = a.sentiment.toLowerCase();
  const badgeClass = `badge-${sentClass}`;
  const modalities = a.modalities_used || [];

  // Suggestions
  const suggestionsHtml = (a.suggestions || []).map(
    (s) => `<div class="suggestion-item">${escapeHtml(s)}</div>`
  ).join("");

  return `
    <div class="analysis-card">
      <div class="analysis-header">
        <h3>📊 Analysis Report</h3>
        <span class="analysis-badge ${badgeClass}">${a.sentiment}</span>
      </div>
      <div class="analysis-grid">
        <div class="analysis-metric">
          <div class="metric-label">Sentiment</div>
          <div class="metric-value ${sentClass}">${a.sentiment}</div>
        </div>
        <div class="analysis-metric">
          <div class="metric-label">Confidence</div>
          <div class="metric-value">${a.confidence.toFixed(0)}%</div>
          <div class="confidence-bar-container">
            <div class="confidence-bar">
              <div class="confidence-fill ${sentClass}" style="width: ${a.confidence}%"></div>
            </div>
          </div>
        </div>
      </div>
      ${suggestionsHtml ? `
      <div class="suggestions-section">
        <div class="suggestions-title">💡 Recommended Actions</div>
        ${suggestionsHtml}
      </div>` : ""}
    </div>`;
}

// Tab switching
window.switchTab = function (el, modality) {
  const card = el.closest(".analysis-card");
  card.querySelectorAll(".modality-tab").forEach((t) => t.classList.remove("active"));
  card.querySelectorAll(".modality-detail").forEach((p) => p.classList.remove("active"));
  el.classList.add("active");
  card.querySelector(`[data-panel="${modality}"]`).classList.add("active");
};

// ── Send message ───────────────────────────────────────────────────
async function handleSend() {
  if (isProcessing) return;
  const text = messageInput.value.trim();
  if (!text && !pendingAudio && !pendingImage) return;

  // Ensure a conversation exists
  if (!activeConvId) newConversation();
  const conv = conversations.find((c) => c.id === activeConvId);

  // Build user message
  const userMsg = {
    role: "user",
    content: text || "(file upload)",
    time: timeNow(),
    audioName: pendingAudio?.name || null,
    imageName: pendingImage?.name || null,
  };
  conv.messages.push(userMsg);

  // Update title from first message
  if (conv.messages.length === 1 && text) {
    conv.title = text.length > 40 ? text.slice(0, 40) + "…" : text;
    renderConversationList();
  }

  renderMessages();
  messageInput.value = "";
  autoResize();

  // Show typing indicator
  isProcessing = true;
  sendBtn.disabled = true;
  appendTypingIndicator();

  // Build FormData
  const fd = new FormData();
  if (text) fd.append("text", text);
  if (pendingAudio) fd.append("audio", pendingAudio);
  if (pendingImage) fd.append("image", pendingImage);

  pendingAudio = null;
  pendingImage = null;
  updateFilePreview();

  try {
    const res = await fetch(`${API}/analyse`, { method: "POST", body: fd });
    removeTypingIndicator();

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Server error" }));
      addAssistantMessage(`⚠️ Error: ${err.detail || res.statusText}`, null, conv);
    } else {
      const data = await res.json();
      const summaryParts = [];
      summaryParts.push(`I detected **${data.sentiment}** sentiment (${data.confidence.toFixed(0)}% confidence).`);

      const modLabels = data.modalities_used.map((m) => m.charAt(0).toUpperCase() + m.slice(1));
      summaryParts.push(`Modalities analysed: ${modLabels.join(", ")}.`);

      addAssistantMessage(summaryParts.join(" "), data, conv);
    }
  } catch (e) {
    removeTypingIndicator();
    addAssistantMessage(`⚠️ Network error: ${e.message}`, null, conv);
  }

  isProcessing = false;
  sendBtn.disabled = false;
}

function addAssistantMessage(content, analysis, conv) {
  // Clean markdown-style bold for plain display
  const plainContent = content.replace(/\*\*(.*?)\*\*/g, "$1");
  conv.messages.push({
    role: "assistant",
    content: plainContent,
    analysis: analysis,
    time: timeNow(),
  });
  renderMessages();
}

// ── Typing indicator ───────────────────────────────────────────────
function appendTypingIndicator() {
  const div = document.createElement("div");
  div.className = "message assistant typing-msg";
  div.innerHTML = `
    <div class="message-avatar">AI</div>
    <div class="message-body">
      <div class="message-header">
        <span class="message-name">Sentiment AI</span>
      </div>
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  messagesContainer.appendChild(div);
  scrollToBottom();
}

function removeTypingIndicator() {
  const t = $(".typing-msg");
  if (t) t.remove();
}

// ── Helpers ────────────────────────────────────────────────────────
function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function timeNow() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function escapeHtml(str) {
  if (!str) return "";
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
