/**
 * RAG Chat Interface - Adapted from original beautiful design
 */

class RAGChat {
    constructor() {
        this.currentModel = 'mistral';
        this.isTyping = false;
        this.sessionId = window.APP_CONFIG?.sessionId || '';
        this.elements = this.initializeElements();
        this.init();
    }

    initializeElements() {
        return {
            messageInput: document.getElementById('message-input'),
            messagesContainer: document.getElementById('messages'),
            modelSelect: document.getElementById('model-select'),
            typingIndicator: document.getElementById('typing-indicator'),
            notificationsContainer: document.getElementById('notifications'),
            clearBtn: document.getElementById('clear-btn'),
            refreshDocsBtn: document.getElementById('refresh-docs-btn'),
            confirmModal: document.getElementById('confirm-modal'),
            confirmOk: document.getElementById('confirm-ok'),
            confirmCancel: document.getElementById('confirm-cancel'),
            gpuStatus: document.getElementById('gpu-status'),
            docsStatus: document.getElementById('docs-status'),
            memoryStatus: document.getElementById('memory-status'),
            llmStatus: document.getElementById('llm-status')
        };
    }

    init() {
        this.setupEventListeners();
        this.loadConversation();
        this.checkSystemStatus();
        this.startStatusCheck();
    }

    setupEventListeners() {
        const { messageInput, modelSelect, clearBtn, refreshDocsBtn } = this.elements;

        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        messageInput.addEventListener('input', () => {
            messageInput.style.height = 'auto';
            messageInput.style.height = Math.min(messageInput.scrollHeight, 150) + 'px';
        });

        modelSelect.addEventListener('change', (e) => {
            this.currentModel = e.target.value;
            this.showNotification(`Модель изменена на ${e.target.value}`, 'info');
        });

        clearBtn.addEventListener('click', () => this.handleClearClick());
        refreshDocsBtn.addEventListener('click', () => this.refreshDocuments());
    }

    async sendMessage() {
        const { messageInput } = this.elements;
        const message = messageInput.value.trim();
        
        if (!message || this.isTyping) return;

        try {
            this.addMessage(message, 'user');
            messageInput.value = '';
            messageInput.style.height = 'auto';
            this.showTypingIndicator();

            const response = await fetch('/send_message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    model: this.currentModel
                })
            });

            const data = await response.json();

            if (data.success) {
                this.addMessage(data.response, 'ai', data.response_time, data.context_info);
                this.updateMemoryStatus();
            } else {
                this.addMessage(`❌ ${data.error}`, 'ai');
                this.showNotification(`Ошибка: ${data.error}`, 'error');
            }
        } catch (error) {
            this.addMessage(`❌ Ошибка соединения: ${error.message}`, 'ai');
            this.showNotification('Ошибка соединения', 'error');
        } finally {
            this.hideTypingIndicator();
        }
    }

    addMessage(content, type, responseTime = null, contextInfo = null) {
        const { messagesContainer } = this.elements;
        
        // Remove welcome message
        const welcomeMessage = messagesContainer.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        let contextHTML = '';
        if (type === 'ai' && contextInfo) {
            const { context_used, documents_found, memory_context_found, gpu_enabled } = contextInfo;
            if (context_used) {
                contextHTML = `
                    <div class="context-info">
                        🧠 Контекст: ${documents_found} док., ${memory_context_found || 0} воспоминаний
                        ${gpu_enabled ? ' 🚀' : ' 💻'}
                    </div>
                `;
            } else {
                contextHTML = `<div class="context-info">💭 Без контекста ${gpu_enabled ? ' 🚀' : ' 💻'}</div>`;
            }
        }
        
        let timeText = new Date().toLocaleTimeString();
        if (responseTime) {
            timeText += ` (${responseTime.toFixed(1)}s)`;
        }
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${type === 'user' ? '👤' : '🤖'}</div>
            <div class="message-content">
                ${this.processContent(content)}
                ${contextHTML}
                <div class="message-time">${timeText}</div>
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        // Highlight code
        if (typeof Prism !== 'undefined') {
            Prism.highlightAllUnder(messageDiv);
        }
    }

    processContent(content) {
        return content
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px;">$1</code>')
            .replace(/\n/g, '<br>');
    }

    showTypingIndicator() {
        this.isTyping = true;
        this.elements.typingIndicator.style.display = 'flex';
    }

    hideTypingIndicator() {
        this.isTyping = false;
        this.elements.typingIndicator.style.display = 'none';
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        this.elements.notificationsContainer.appendChild(notification);
        setTimeout(() => notification.remove(), 3000);
    }

    async loadConversation() {
        try {
            const response = await fetch('/get_conversation');
            const conversation = await response.json();
            
            conversation.forEach(item => {
                this.addMessage(item.user, 'user');
                this.addMessage(item.assistant, 'ai', item.response_time, {
                    context_used: item.context_used,
                    documents_found: item.documents_found,
                    gpu_enabled: true
                });
            });
        } catch (error) {
            console.error('Error loading conversation:', error);
        }
    }

    async clearConversation() {
        try {
            await fetch('/clear_conversation', { method: 'POST' });
            this.elements.messagesContainer.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-avatar">🤖</div>
                    <div class="welcome-content">
                        <h3>Чат очищен!</h3>
                        <p>Готов к новому разговору с RAG функциональностью!</p>
                    </div>
                </div>
            `;
            this.updateMemoryStatus();
            this.showNotification('Чат и память очищены', 'success');
        } catch (error) {
            this.showNotification('Ошибка очистки чата', 'error');
        }
    }

    async refreshDocuments() {
        const { refreshDocsBtn } = this.elements;
        const originalHTML = refreshDocsBtn.innerHTML;
        
        refreshDocsBtn.disabled = true;
        refreshDocsBtn.innerHTML = '🔄 Обновление...';
        
        try {
            const response = await fetch('/refresh_documents', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(data.message, 'success');
                this.checkSystemStatus();
            } else {
                this.showNotification(`Ошибка: ${data.error}`, 'error');
            }
        } catch (error) {
            this.showNotification('Ошибка обновления документов', 'error');
        } finally {
            refreshDocsBtn.disabled = false;
            refreshDocsBtn.innerHTML = originalHTML;
        }
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/check_status');
            const data = await response.json();
            this.updateStatus(data);
        } catch (error) {
            console.error('Status check error:', error);
        }
    }

    updateStatus(data) {
        const { gpuStatus, docsStatus, llmStatus, modelSelect } = this.elements;
        
        // GPU status
        if (data.gpu_info?.gpu_available) {
            gpuStatus.textContent = `GPU: ✅ ${data.gpu_info.gpu_name || 'RTX 4060'}`;
        } else {
            gpuStatus.textContent = 'GPU: ❌ Недоступно';
        }
        
        // Documents status
        const docs = data.documents_loaded;
        docsStatus.textContent = `Документы: 📚 ${docs.total_chunks} из ${docs.unique_files} файлов`;
        
        // LLM status
        if (data.llm_available) {
            llmStatus.textContent = `LLM: ✅ Доступно (${data.available_models.length} моделей)`;
            
            // Update model select
            modelSelect.innerHTML = '';
            data.available_models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                if (model === this.currentModel) option.selected = true;
                modelSelect.appendChild(option);
            });
        } else {
            llmStatus.textContent = 'LLM: ❌ Недоступно';
        }
        
        this.updateMemoryStatus();
    }

    async updateMemoryStatus() {
        try {
            const response = await fetch('/memory_stats');
            const stats = await response.json();
            this.elements.memoryStatus.textContent = `Память: 💾 ${stats.total_messages || 0} сообщений`;
        } catch (error) {
            console.error('Error updating memory status:', error);
        }
    }

    startStatusCheck() {
        setInterval(() => this.checkSystemStatus(), 30000);
    }

    handleClearClick() {
        const { confirmModal, confirmOk, confirmCancel } = this.elements;
        confirmModal.classList.add('show');

        const close = () => confirmModal.classList.remove('show');
        
        const okHandler = () => { 
            this.clearConversation(); 
            detach(); 
        };
        const cancelHandler = () => detach();
        
        function detach() {
            confirmOk.removeEventListener('click', okHandler);
            confirmCancel.removeEventListener('click', cancelHandler);
            close();
        }
        
        confirmOk.addEventListener('click', okHandler);
        confirmCancel.addEventListener('click', cancelHandler);
    }
}

// Initialize the RAG chat
document.addEventListener('DOMContentLoaded', () => {
    new RAGChat();
}); 