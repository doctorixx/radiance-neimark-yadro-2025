/**
 * Modern AI Chat Interface
 * Clean JavaScript with ES6+ features and proper error handling
 */

class AIChat {
    constructor() {
        this.currentModel = 'qwen2.5:7b';
        this.isTyping = false;
        this.abortController = null;
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
            stopBtn: document.getElementById('stop-btn'),
            predictionsContainer: document.getElementById('predictions-container'),
            predictionsList: document.getElementById('predictions-list'),
            closePredictionsBtn: document.getElementById('close-predictions'),
            confirmModal: document.getElementById('confirm-modal'),
            confirmMessage: document.getElementById('confirm-message'),
            confirmOk: document.getElementById('confirm-ok'),
            confirmCancel: document.getElementById('confirm-cancel')
        };
    }

    init() {
        this.setupEventListeners();
        this.loadConversation();
        this.autoResizeTextarea();
        this.startStatusCheck();
    }

    setupEventListeners() {
        const { messageInput, modelSelect, clearBtn, stopBtn, closePredictionsBtn, predictionsContainer } = this.elements;

        // Send message events
        messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        messageInput.addEventListener('input', () => this.autoResizeTextarea());

        // Control events
        modelSelect.addEventListener('change', (e) => this.changeModel(e.target.value));

        // Clear conversation button
        clearBtn.addEventListener('click', () => this.handleClearClick());
        
        // Stop generation button
        stopBtn.addEventListener('click', () => this.stopGeneration());
        
        // Close predictions button
        closePredictionsBtn.addEventListener('click', () => this.hidePredictions());
        
        // Close predictions on Esc key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && predictionsContainer.style.display === 'block') {
                this.hidePredictions();
            }
        });
        
        // Close predictions on click outside
        document.addEventListener('click', (e) => {
            if (predictionsContainer.style.display === 'block' && 
                !predictionsContainer.contains(e.target)) {
                this.hidePredictions();
            }
        });
    }

    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }

    autoResizeTextarea() {
        const { messageInput } = this.elements;
        messageInput.style.height = 'auto';
        messageInput.style.height = `${Math.min(messageInput.scrollHeight, 150)}px`;
    }

    changeModel(model) {
        this.currentModel = model;
        this.elements.modelSelect.title = this.getModelDescription(model);
        this.showNotification(`Model changed to ${model}`, 'info');
    }

    async sendMessage() {
        const { messageInput } = this.elements;
        const message = messageInput.value.trim();
        
        if (!message || this.isTyping) return;

        try {
            this.addMessage(message, 'user');
            messageInput.value = '';
            this.autoResizeTextarea();
            this.showTypingIndicator();
            this.showStopButton();

            // Создаём новый AbortController для этого запроса
            this.abortController = new AbortController();

            const response = await this.callAPI('/send_message', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    model: this.currentModel
                })
            }, this.abortController.signal);

            if (response.success) {
                this.addMessage(response.response, 'ai', response.response_time, response.message_id);
                // Загружаем новые предсказания после получения ответа
                this.loadPredictions();
            } else {
                this.addMessage(`❌ ${response.error}`, 'ai');
                this.showNotification(`Error: ${response.error}`, 'error');
            }
        } catch (error) {
            if (error.name === 'AbortError') {
                this.addMessage('⏹️ Генерация остановлена пользователем', 'ai');
                this.showNotification('Генерация остановлена', 'info');
            } else {
                this.addMessage(`❌ Connection error: ${error.message}`, 'ai');
                this.showNotification('Connection error', 'error');
            }
        } finally {
            this.hideTypingIndicator();
            this.hideStopButton();
            this.abortController = null;
        }
    }

    addMessage(content, type, responseTime = null, messageId = null, reaction = null) {
        const { messagesContainer } = this.elements;
        
        const messageElement = this.createMessageElement(content, type, responseTime, messageId, reaction);
        messagesContainer.appendChild(messageElement);
        
        this.scrollToBottom();
        this.highlightCode(messageElement);
        this.addCopyHandlers(messageElement);
        
        // Добавляем обработчики реакций только для AI сообщений
        if (type === 'ai' && messageId) {
            this.addReactionHandlers(messageElement, messageId);
        }
    }

    createMessageElement(content, type, responseTime, messageId = null, reaction = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const reactionButtons = type === 'ai' && messageId ? `
            <div class="reaction-buttons" data-message-id="${messageId}">
                <button class="reaction-btn like-btn ${reaction === 'like' ? 'active' : ''}" 
                        data-reaction="like" title="Нравится">
                    👍
                </button>
                <button class="reaction-btn dislike-btn ${reaction === 'dislike' ? 'active' : ''}" 
                        data-reaction="dislike" title="Не нравится">
                    👎
                </button>
            </div>
        ` : '';
        
        messageDiv.innerHTML = `
            <div class="message-avatar">${type === 'user' ? '👤' : '🤖'}</div>
            <div class="message-content">
                ${this.processContent(content)}
                <div class="message-footer">
                    <div class="message-time">
                        ${this.formatTime(responseTime)}
                    </div>
                    ${reactionButtons}
                </div>
            </div>
        `;

        return messageDiv;
    }

    processContent(content) {
        // Разбиваем сообщение на сегменты: текст и блоки кода
        const parts = content.split(/```/);
        let html = '';
        parts.forEach((part, idx) => {
            if (idx % 2 === 0) {
                // Обычный текст: обрабатываем заголовки, inline-code и переносы строк
                let segment = part
                    // Обработка заголовков
                    .replace(/^#### (.+)$/gm, '<h4 style="color: #f0f0f0; margin: 10px 0 5px 0; font-size: 1.1rem; font-weight: 600;">$1</h4>')
                    .replace(/^### (.+)$/gm, '<h3 style="color: #f0f0f0; margin: 12px 0 6px 0; font-size: 1.15rem; font-weight: 600;">$1</h3>')
                    .replace(/^## (.+)$/gm, '<h2 style="color: #f0f0f0; margin: 15px 0 8px 0; font-size: 1.25rem; font-weight: 600;">$1</h2>')
                    .replace(/^# (.+)$/gm, '<h1 style="color: #f0f0f0; margin: 15px 0 10px 0; font-size: 1.4rem; font-weight: 700;">$1</h1>')
                    // Жирный текст (**text**)
                    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                    // Жирный текст в скобках
                    .replace(/\(([^)]+)\)/g, '(<strong>$1</strong>)')
                    // Inline code
                    .replace(/`([^`]+)`/g, '<code style="background: rgba(255,255,255,0.1); padding: 2px 6px; border-radius: 4px; font-family: monospace;">$1</code>')
                    // Переносы строк
                    .replace(/\n/g, '<br>');
                html += segment;
            } else {
                // Блок кода
                let lang = 'text';
                let codeBody = part;
                const newlineIdx = part.indexOf('\n');
                if (newlineIdx !== -1) {
                    lang = part.slice(0, newlineIdx).trim() || 'text';
                    codeBody = part.slice(newlineIdx + 1);
                }
                const escapedCode = this.preserveCodeFormatting(codeBody);
                html += `
                    <div class="code-block">
                        <div class="code-header">
                            <span class="code-language">${lang}</span>
                            <button class="copy-btn" title="Copy" data-code="${this.escapeForAttribute(codeBody)}">📋</button>
                        </div>
                        <pre><code class="language-${lang}">${escapedCode}</code></pre>
                    </div>
                `;
            }
        });
        return html;
    }

    preserveCodeFormatting(code) {
        // Экранируем HTML символы, НО сохраняем все пробелы и табы
        return code
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
        // НЕ трогаем пробелы, табы и переносы строк!
    }

    escapeForAttribute(text) {
        // Для атрибутов HTML нужно другое экранирование
        return text
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    formatTime(responseTime) {
        const now = new Date();
        let timeText = now.toLocaleTimeString();
        if (responseTime) {
            timeText += ` (${responseTime.toFixed(1)}s)`;
        }
        return timeText;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    addCopyHandlers(messageElement) {
        const copyBtns = messageElement.querySelectorAll('.copy-btn');
        copyBtns.forEach(btn => {
            btn.addEventListener('click', async () => {
                const code = btn.getAttribute('data-code');
                await this.copyToClipboard(code);
                this.showCopyFeedback(btn);
            });
        });
    }

    addReactionHandlers(messageElement, messageId) {
        const reactionBtns = messageElement.querySelectorAll('.reaction-btn');
        reactionBtns.forEach(btn => {
            btn.addEventListener('click', async () => {
                const reaction = btn.getAttribute('data-reaction');
                const wasActive = btn.classList.contains('active');
                
                // Деактивировать все кнопки реакций в этом сообщении
                const allReactionBtns = messageElement.querySelectorAll('.reaction-btn');
                allReactionBtns.forEach(b => b.classList.remove('active'));
                
                // Если кнопка не была активной, активируем её
                const finalReaction = wasActive ? null : reaction;
                if (!wasActive) {
                    btn.classList.add('active');
                }
                
                // Отправляем реакцию на сервер
                await this.sendReaction(messageId, finalReaction);
            });
        });
    }

    async sendReaction(messageId, reaction) {
        try {
            const response = await this.callAPI('/add_reaction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message_id: messageId,
                    reaction: reaction
                })
            });

            if (!response.success) {
                console.error('Failed to send reaction:', response.error);
                this.showNotification('Ошибка при отправке реакции', 'error');
            }
        } catch (error) {
            console.error('Error sending reaction:', error);
            this.showNotification('Ошибка при отправке реакции', 'error');
        }
    }

    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            // Убираю уведомление
        } catch (err) {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            // Убираю уведомление
        }
    }

    showCopyFeedback(button) {
        const originalText = button.innerHTML;
        button.innerHTML = '✅';
        button.style.backgroundColor = '#4ade80';
        button.style.color = '#1a1a1a';

        setTimeout(() => {
            button.innerHTML = originalText;
            button.style.backgroundColor = '';
            button.style.color = '';
        }, 1000);
    }

    showTypingIndicator() {
        this.isTyping = true;
        this.elements.typingIndicator.style.display = 'flex';
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        this.elements.typingIndicator.style.display = 'none';
    }

    showStopButton() {
        this.elements.stopBtn.style.display = 'flex';
    }

    hideStopButton() {
        this.elements.stopBtn.style.display = 'none';
    }

    stopGeneration() {
        if (this.abortController) {
            this.abortController.abort();
            this.showNotification('Остановка генерации...', 'info');
        }
    }

    scrollToBottom() {
        const { messagesContainer } = this.elements;
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        this.elements.notificationsContainer.appendChild(notification);
        
        setTimeout(() => notification.remove(), 3000);
    }

    highlightCode(element) {
        if (typeof Prism !== 'undefined') {
            Prism.highlightAllUnder(element);
        }
    }

    async loadConversation() {
        try {
            const conversation = await this.callAPI('/get_conversation');
            
            conversation.forEach(item => {
                this.addMessage(item.user, 'user');
                this.addMessage(item.assistant, 'ai', item.response_time, item.message_id, item.reaction);
            });
            
            // Загружаем предсказания после загрузки диалога
            this.loadPredictions();
        } catch (error) {
            console.error('Error loading conversation:', error);
        }
    }

    async loadPredictions() {
        try {
            console.log('🔮 Загружаем предсказания...');
            const response = await this.callAPI('/get_predictions');
            console.log('📡 Ответ сервера:', response);
            
            if (response.predictions && response.predictions.length > 0) {
                console.log('✅ Показываем предсказания:', response.predictions);
                this.showPredictions(response.predictions);
            } else {
                console.log('❌ Нет предсказаний, скрываем блок');
                this.hidePredictions();
            }
        } catch (error) {
            console.error('❌ Error loading predictions:', error);
            this.hidePredictions();
        }
    }

    showPredictions(predictions) {
        const { predictionsContainer, predictionsList } = this.elements;
        
        // Очищаем предыдущие предсказания
        predictionsList.innerHTML = '';
        
        // Типы вопросов с стилями (без эмодзи)
        const questionTypes = [
            { class: 'research', title: 'Исследовательский вопрос' },
            { class: 'practical', title: 'Практический вопрос' },
            { class: 'detailed', title: 'Подробный вопрос' },
            { class: 'quick', title: 'Быстрый вопрос' },
            { class: 'developing', title: 'Развивающий вопрос' }
        ];
        
        // Создаём чипы для каждого предсказания
        predictions.forEach((prediction, index) => {
            const chip = document.createElement('div');
            const type = questionTypes[index] || questionTypes[0];
            
            chip.className = `prediction-chip prediction-${type.class}`;
            chip.title = type.title;
            chip.textContent = prediction;
            
            chip.addEventListener('click', () => {
                this.selectPrediction(prediction);
            });
            predictionsList.appendChild(chip);
        });
        
        // Показываем контейнер
        predictionsContainer.style.display = 'block';
    }

    hidePredictions() {
        this.elements.predictionsContainer.style.display = 'none';
    }

    selectPrediction(prediction) {
        const { messageInput } = this.elements;
        
        // Вставляем предсказание в поле ввода
        messageInput.value = prediction;
        messageInput.focus();
        this.autoResizeTextarea();
        
        // Скрываем предсказания после выбора
        this.hidePredictions();
        
        // Можно автоматически отправить сообщение или оставить пользователю возможность редактировать
        // this.sendMessage(); // Раскомментировать для автоотправки
    }

    async clearConversation() {
        // подтверждение происходит через кастомный модал
        try {
            await this.callAPI('/clear_conversation', { method: 'POST' });
            this.elements.messagesContainer.innerHTML = '';
            this.hidePredictions(); // Скрываем предсказания при очистке
            this.showNotification('Conversation cleared', 'success');
        } catch (error) {
            this.showNotification('Error clearing conversation', 'error');
        }
    }

    async exportConversation() {
        try {
            const conversation = await this.callAPI('/get_conversation');
            
            if (conversation.length === 0) {
                this.showNotification('No messages to export', 'info');
                return;
            }
            
            const exportText = this.generateExportText(conversation);
            this.downloadFile(exportText, `chat_export_${new Date().toISOString().split('T')[0]}.txt`);
            
            this.showNotification('Chat exported', 'success');
        } catch (error) {
            this.showNotification('Export error', 'error');
        }
    }

    generateExportText(conversation) {
        let text = `AI Chat Export - ${new Date().toLocaleString()}\n`;
        text += '='.repeat(50) + '\n\n';
        
        conversation.forEach((item, index) => {
            text += `Message ${index + 1}:\n`;
            text += `Time: ${new Date(item.timestamp).toLocaleString()}\n`;
            text += `User: ${item.user}\n`;
            text += `AI (${item.response_time.toFixed(1)}s): ${item.assistant}\n\n`;
            text += '-'.repeat(30) + '\n\n';
        });
        
        return text;
    }

    downloadFile(content, filename) {
        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }

    async checkStatus() {
        try {
            const data = await this.callAPI('/check_status');
            this.updateStatus(data);
        } catch (error) {
            console.error('Status check error:', error);
        }
    }

    updateStatus(data) {
        const statusElement = document.querySelector('.status span');
        if (statusElement) {
            if (data.ollama_running) {
                statusElement.textContent = '🟢 Ollama running';
                statusElement.className = 'status-online';
            } else {
                statusElement.textContent = '�� Ollama not running';
                statusElement.className = 'status-offline';
            }
        }
        this.updateModelSelect(data.models);
    }

    getModelDescription(modelName) {
        const descriptions = {
            'qwen2.5:7b': 'Средняя модель - Хороший баланс скорости и качества (32K токенов)',
            'qwen2.5:14b': 'Умная модель - Лучше понимает контекст и дает качественные ответы (32K токенов)',
            'llama3.1:8b': 'Огромный контекст - 128K токенов! Отлично для длинных документов'
        };
        
        return descriptions[modelName] || 'AI модель';
    }

    updateModelSelect(models) {
        const { modelSelect } = this.elements;
        modelSelect.innerHTML = '';
        
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            option.title = this.getModelDescription(model);
            if (model === this.currentModel) {
                option.selected = true;
            }
            modelSelect.appendChild(option);
        });
        
        // Добавляем tooltip к самому select элементу
        modelSelect.title = this.getModelDescription(this.currentModel);
    }

    startStatusCheck() {
        // Check status every 30 seconds
        setInterval(() => this.checkStatus(), 30000);
    }

    async callAPI(endpoint, options = {}, signal = null) {
        const fetchOptions = { ...options };
        if (signal) {
            fetchOptions.signal = signal;
        }
        
        const response = await fetch(endpoint, fetchOptions);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    }

    handleClearClick() {
        this.showConfirmation('Уверены, что хотите очистить чат?', 'Очистить', () => this.clearConversation());
    }

    showConfirmation(message, confirmLabel = 'OK', onConfirm = () => {}) {
        const { confirmModal, confirmMessage, confirmOk, confirmCancel } = this.elements;
        confirmMessage.textContent = message;
        confirmOk.textContent = confirmLabel;
        confirmModal.classList.add('show');

        const close = () => confirmModal.classList.remove('show');
        const okHandler = () => { onConfirm(); detach(); };
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

// Initialize the chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AIChat();
}); 