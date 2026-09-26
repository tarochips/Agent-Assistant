(function () {
    const API = "/api";

    const state = {
        sessions: [],
        activeSessionId: null,
        streamingAbort: null,
    };

    const $ = (selector) => document.querySelector(selector);
    const sessionList = $("#session-list");
    const messagesEl = $("#messages");
    const userInput = $("#user-input");
    const btnSend = $("#btn-send");
    const btnNew = $("#btn-new-session");
    const sessionTitle = $("#current-session-title");
    const docList = $("#doc_list");
    const fileInput = $("#file-input");
    const uploadStatus = $("#upload-status");

    function errorMessage(payload, fallback = "请求失败") {
        if (!payload) return fallback;
        if (typeof payload.error === "string") return payload.error;
        if (payload.error && payload.error.message) return payload.error.message;
        if (typeof payload.detail === "string") return payload.detail;
        return fallback;
    }

    async function api(method, path, body) {
        const options = { method, headers: {} };
        if (body !== undefined) {
            options.headers["Content-Type"] = "application/json";
            options.body = JSON.stringify(body);
        }
        const response = await fetch(API + path, options);
        const payload = await response.json().catch(() => null);
        if (!response.ok) throw new Error(errorMessage(payload));
        return payload;
    }

    async function loadSessions() {
        state.sessions = await api("GET", "/sessions");
        renderSessions();
    }

    function renderSessions() {
        sessionList.innerHTML = "";
        state.sessions.forEach((session) => {
            const item = document.createElement("li");
            item.dataset.id = session.id;
            if (session.id === state.activeSessionId) item.classList.add("activeSessionId");

            const title = document.createElement("span");
            title.className = "session-title";
            title.textContent = session.title || "未命名";
            title.onclick = () => switchSession(session.id);

            const remove = document.createElement("span");
            remove.className = "del-session";
            remove.textContent = "x";
            remove.onclick = async (event) => {
                event.stopPropagation();
                await deleteSession(session.id);
            };

            item.append(title, remove);
            sessionList.appendChild(item);
        });
    }

    async function switchSession(id) {
        state.activeSessionId = id;
        const session = await api("GET", `/sessions/${id}`);
        messagesEl.innerHTML = "";
        session.messages.forEach((message) => {
            const content = addMessage(message.role, message.content);
            if (message.sources && message.sources.length) {
                renderSources(content, message.sources);
            }
        });
        sessionTitle.textContent = session.title;
        btnSend.disabled = false;
        renderSessions();
        scrollBottom();
    }

    async function deleteSession(id) {
        await api("DELETE", `/sessions/${id}`);
        if (state.activeSessionId === id) {
            state.activeSessionId = null;
            messagesEl.innerHTML = "";
            sessionTitle.textContent = "选择一个会话";
            btnSend.disabled = true;
        }
        await loadSessions();
    }

    async function loadDocuments() {
        const documents = await api("GET", "/documents");
        docList.innerHTML = "";
        documents.forEach((doc) => {
    const item = document.createElement("li");

    const label = document.createElement("span");
    label.textContent = `${doc.filename} (${doc.chunk_count}块)`;

    const remove = document.createElement("span");
    remove.className = "del-doc";
    remove.textContent = "x";
    remove.onclick = async () => {
        await api("DELETE", `/documents/${doc.doc_id}`);
        await loadDocuments();
    };

    item.append(label, remove);
    docList.appendChild(item);
});
    }

    fileInput.addEventListener("change", async () => {
        const file = fileInput.files[0];
        if (!file) return;
        uploadStatus.textContent = "上传中...";
        const form = new FormData();
        form.append("file", file);

        try {
            const response = await fetch(`${API}/documents/upload`, {
                method: "POST",
                body: form,
            });
            const payload = await response.json().catch(() => null);
            if (!response.ok) throw new Error(errorMessage(payload, "上传失败"));
            uploadStatus.textContent = "上传完成";
            await loadDocuments();
        } catch (error) {
            uploadStatus.textContent = error.message;
        } finally {
            fileInput.value = "";
            setTimeout(() => { uploadStatus.textContent = ""; }, 3000);
        }
    });

    function addMessage(role, content, idHint) {
        const message = document.createElement("div");
        message.className = role === "user"
            ? "message user-message"
            : "message assistant-message";

        const body = document.createElement("div");
        body.className = "message-body";

        const contentEl = document.createElement("div");
        contentEl.className = "message-content";
        contentEl.textContent = content;
        if (idHint) contentEl.id = idHint;

        body.appendChild(contentEl);
        message.appendChild(body);
        messagesEl.appendChild(message);
        scrollBottom();
        return contentEl;
    }

    function renderSources(contentEl, sources) {
        const body = contentEl.parentElement;
        const existing = body.querySelector(".message-sources");
        if (existing) existing.remove();
        if (!sources || !sources.length) return;

        const details = document.createElement("details");
        details.className = "message-sources";
        const summary = document.createElement("summary");
        summary.textContent = `参考来源 (${sources.length})`;
        details.appendChild(summary);

        const list = document.createElement("ol");
        sources.forEach((source) => {
            const item = document.createElement("li");
            const heading = document.createElement("strong");
            heading.textContent = `${source.filename} · 分块 ${source.chunk_index + 1}`;
            const score = document.createElement("span");
            score.className = "source-score";
            score.textContent = `相关度 ${Number(source.score).toFixed(3)}`;
            const excerpt = document.createElement("p");
            excerpt.textContent = source.excerpt;
            item.append(heading, score, excerpt);
            list.appendChild(item);
        });
        details.appendChild(list);
        body.appendChild(details);
    }

    function scrollBottom() {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    btnNew.addEventListener("click", async () => {
        const session = await api("POST", "/sessions");
        state.activeSessionId = session.id;
        messagesEl.innerHTML = "";
        sessionTitle.textContent = session.title;
        btnSend.disabled = false;
        await loadSessions();
    });

    function parseSseBlock(block) {
        let eventName = "message";
        const dataLines = [];
        block.split("\n").forEach((line) => {
            if (line.startsWith("event:")) eventName = line.slice(6).trim();
            if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
        });
        if (!dataLines.length) return null;
        const raw = dataLines.join("\n");
        if (!raw || raw === "[DONE]") return null;
        return { eventName, payload: JSON.parse(raw) };
    }

    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        if (!state.activeSessionId) {
            const session = await api("POST", "/sessions");
            state.activeSessionId = session.id;
            sessionTitle.textContent = session.title;
            await loadSessions();
        }

        addMessage("user", query);
        userInput.value = "";
        userInput.style.height = "auto";
        const botContent = addMessage("assistant", "思考中", "stream");
        btnSend.disabled = true;

        const controller = new AbortController();
        state.streamingAbort = controller;
        let started = false;
        let completed = false;
        let sources = [];

        try {
            const response = await fetch(`${API}/chat/stream`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    message: query,
                    session_id: state.activeSessionId,
                }),
                signal: controller.signal,
            });
            if (!response.ok) {
                const payload = await response.json().catch(() => null);
                throw new Error(errorMessage(payload));
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
                const blocks = buffer.split("\n\n");
                buffer = blocks.pop();

                for (const block of blocks) {
                    if (!block.trim()) continue;
                    const parsed = parseSseBlock(block);
                    if (!parsed) continue;
                    const { eventName, payload } = parsed;
                    const type = payload.type || eventName;

                    if (type === "retrieval") {
                        sources = payload.sources || [];
                        if (payload.session_id) state.activeSessionId = payload.session_id;
                    } else if (type === "token") {
                        if (!started) {
                            botContent.textContent = "";
                            started = true;
                        }
                        botContent.textContent += payload.delta;
                        scrollBottom();
                    } else if (type === "completed") {
                        completed = true;
                        sources = payload.citations || sources;
                        renderSources(botContent, sources);
                    } else if (type === "error") {
                        botContent.textContent = `错误：${payload.message}`;
                        started = true;
                    }
                }
            }

            if (!started && completed) botContent.textContent = "模型未返回内容";
            if (!started && !completed) botContent.textContent = "回答已终止";
        } catch (error) {
            botContent.textContent = error.name === "AbortError"
                ? "回答已终止"
                : `请求失败：${error.message}`;
        } finally {
            state.streamingAbort = null;
            btnSend.disabled = false;
            botContent.removeAttribute("id");
            await loadSessions();
        }
    }

    btnSend.addEventListener("click", sendMessage);
    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (!btnSend.disabled) sendMessage();
        }
    });
    userInput.addEventListener("input", () => {
        userInput.style.height = "auto";
        userInput.style.height = `${Math.min(userInput.scrollHeight, 200)}px`;
    });

    async function init() {
        try {
            await Promise.all([loadSessions(), loadDocuments()]);
        } catch (error) {
            messagesEl.textContent = `初始化失败：${error.message}`;
        }
    }

    init();
})();
