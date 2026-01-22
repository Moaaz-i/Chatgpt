from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import openai
import time
import asyncio
import json
import logging
import uvicorn
from typing import Dict, List
from datetime import datetime
import os
import traceback
from googlesearch import search as google_search

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chatbot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Try to import Farasa with fallback
try:
    from farasa.segmenter import FarasaSegmenter
    from farasa.pos import FarasaPOSTagger
    from farasa.stemmer import FarasaStemmer
    FARASA_AVAILABLE = True
except ImportError:
    FARASA_AVAILABLE = False
    print("Farasa not available. Install with: pip install farasapy")
    # Create dummy classes for fallback
    class FarasaSegmenter:
        def __init__(self, interactive=False):
            pass
        def segment(self, text):
            return text
    class FarasaPOSTagger:
        def __init__(self, interactive=False):
            pass
        def tag(self, text):  # Changed from pos() to tag() to match Farasa API
            return []
    class FarasaStemmer:
        def __init__(self, interactive=False):
            pass
        def stem(self, text):
            return text

class ArabicChatbot:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        self.client = openai.OpenAI(
            api_key=self.api_key,  # or os.getenv("POE_API_KEY")
            base_url = "https://api.poe.com/v1",
        )
        self.history = {}
        logger.info("ArabicChatbot initialized with OpenAI API")

    def initialize_farasa(self):
        """Initialize Farasa components"""
        if not FARASA_AVAILABLE:
            logger.warning("Farasa is not available. Text analysis will be limited.")
        # Rest of the function remains the same

    async def get_response(self, user_id: str, message: str) -> str:
        """Generate response using OpenAI's API with conversation history"""
        try:
            message = message.strip()
            if not message or len(message) < 2:
                return "عذراً، لم أتمكن من فهم رسالتك. هل يمكنك إعادة كتابتها؟"
            
            # Get conversation history
            conversation = self._get_conversation_history(user_id, message)
            
            try:
                # Call OpenAI API
                response = await asyncio.to_thread(
                    self._call_openai_chat,
                    conversation=conversation
                )
                
                # Clean and validate the response
                response = self._clean_response(response)
                if not response or not self._is_valid_response(response):
                    logger.warning(f"Generated invalid response: {response}")
                    return "عذراً، لم أتمكن من معالجة طلبك. هل يمكنك إعادة صياغة سؤالك؟"
                
                # Update conversation history
                self._update_history(user_id, message, response)
                return response
                
            except Exception as e:
                logger.error(f"Error in OpenAI API call: {str(e)}")
                return "عذراً، حدث خطأ أثناء معالجة طلبك. الرجاء المحاولة مرة أخرى."
                
        except Exception as e:
            logger.error(f"Unexpected error in get_response: {str(e)}")
            logger.error(traceback.format_exc())
            return "عذراً، حدث خطأ غير متوقع. الرجاء المحاولة مرة أخرى."

    async def search_web_for_answer(self, query: str) -> str:
        """Search the web for information related to the query"""
        try:
            # Search the web for relevant information
            search_results = list(google_search(
                query,
                num_results=3,  # Limit to 3 results
                lang='ar',      # Prefer Arabic results
                advanced=True   # Get structured results
            ))

            print(search_results)
            
            # Process and format the search results
            if search_results:
                result_text = "\n".join([
                    f"- {result.title}\n  {result.url}"
                    for result in search_results
                    if hasattr(result, 'title') and hasattr(result, 'url')
                ])
                
                if result_text.strip():
                    return (
                        "🔍 هذه بعض النتائج التي وجدتها على الإنترنت:\n\n" +
                        result_text +
                        "\n\nهل تريد المزيد من المعلومات عن أي من هذه النقاط؟"
                    )
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            logger.error(traceback.format_exc())
        
        return "عذراً، لم أتمكن من العثور على معلومات كافية حول هذا الموضوع. هل يمكنك إعادة صياغة سؤالك؟"

    def _call_openai_chat(self, conversation: List[Dict[str, str]]) -> str:
        """Make a call to OpenAI's chat completion API"""
        try:
            response = self.client.chat.completions.create(
                model="assistant",
                messages=conversation,
                temperature=0.7,
                max_tokens=500,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )

            return response.choices[0].message.content.strip(" ")
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _get_conversation_history(self, user_id: str, new_message: str) -> List[Dict[str, str]]:
        """Format the conversation history for the API"""
        messages = [{
            "role": "system",
            "content": "أنت مساعد ذكي يتحدث العربية. كن مفيداً، دقيقاً، وودوداً في إجاباتك. استخدم لغة عربية فصيحة وسهلة الفهم."
        }]
        
        # Add conversation history if available
        if user_id in self.history:
            for msg in self.history[user_id][-5:]:  # Last 5 exchanges
                messages.append({"role": "user", "content": msg['user']})
                messages.append({"role": "assistant", "content": msg['bot']})
        
        # Add the new message
        messages.append({"role": "user", "content": new_message})
        return messages
    
    def _clean_response(self, text: str) -> str:
        """Clean and validate the response from the API"""
        if not text or not isinstance(text, str):
            return ""
            
        # Remove any leading/trailing whitespace
        text = text.strip()
        
        # Remove any empty lines
        lines = [line for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Ensure the response is not too short (optional)
        if len(text) < 3:  # Minimum 3 characters
            return ""
            
        return text
    
    def _is_valid_response(self, text: str) -> bool:
        """Check if the response is valid"""
        if not text or not isinstance(text, str):
            return False
            
        # Check for common error patterns
        error_patterns = [
            "I'm sorry",
            "عذراً",
            "error",
            "خطأ",
            "exception"
        ]
        
        text_lower = text.lower()
        return not any(pattern in text_lower for pattern in error_patterns)
    
    def _update_history(self, user_id: str, user_message: str, bot_response: str):
        """Update the conversation history"""
        if user_id not in self.history:
            self.history[user_id] = []
        
        # Keep only the last 10 exchanges (20 messages total)
        self.history[user_id].append({
            'user': user_message,
            'bot': bot_response,
            'timestamp': datetime.utcnow().isoformat()
        })
        self.history[user_id] = self.history[user_id][-10:]
    
    # Rest of the class remains the same

# Initialize chatbot with your OpenAI API key
chatbot = ArabicChatbot(api_key="H44aCfURvNyky-7Ukk6eb0Yh2c7DhHaFw5cYvGDsEDY")  # Make sure OPENAI_API_KEY is set in your .env file

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, dict] = {}
        self.lock = asyncio.Lock()
        self.ping_interval = 25  # seconds
        self.timeout = 30  # seconds

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        async with self.lock:
            self.active_connections[client_id] = {
                'websocket': websocket,
                'last_ping': time.time(),
                'active': True
            }
            logger.info(f"Client {client_id} connected")
            # Start keep-alive task
            asyncio.create_task(self._keep_alive(client_id))

    async def _keep_alive(self, client_id: str):
        while client_id in self.active_connections and self.active_connections[client_id]['active']:
            try:
                await asyncio.sleep(self.ping_interval)
                if client_id in self.active_connections and self.active_connections[client_id]['active']:
                    websocket = self.active_connections[client_id]['websocket']
                    if websocket.client_state != WebSocketState.DISCONNECTED:
                        await self.send_ping(client_id)
            except Exception as e:
                if client_id in self.active_connections:
                    logger.error(f"Keep-alive error for {client_id}: {str(e)}")
                    await self.disconnect(client_id)
                break

    async def send_ping(self, client_id: str):
        try:
            if client_id in self.active_connections:
                websocket = self.active_connections[client_id]['websocket']
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": time.time()
                    })
                    self.active_connections[client_id]['last_ping'] = time.time()
        except Exception as e:
            if client_id in self.active_connections:
                logger.error(f"Ping failed for {client_id}: {str(e)}")
                await self.disconnect(client_id)

    async def disconnect(self, client_id: str):
        async with self.lock:
            if client_id in self.active_connections:
                try:
                    self.active_connections[client_id]['active'] = False
                    websocket = self.active_connections[client_id]['websocket']
                    if websocket.client_state != WebSocketState.DISCONNECTED:
                        await websocket.close()
                except Exception as e:
                    if 'Cannot call "send"' not in str(e):
                        logger.error(f"Error closing WebSocket for {client_id}: {str(e)}")
                finally:
                    if client_id in self.active_connections:
                        del self.active_connections[client_id]
                    logger.info(f"Client {client_id} disconnected")

    async def send_message(self, message: str, client_id: str) -> bool:
        if client_id not in self.active_connections:
            logger.warning(f"Tried to send message to non-existent client {client_id}")
            return False
            
        try:
            websocket = self.active_connections[client_id]['websocket']
            if websocket.client_state == WebSocketState.DISCONNECTED:
                logger.warning(f"Cannot send message to disconnected client {client_id}")
                return False
                
            await websocket.send_json({
                "type": "message",
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            })
            return True
        except Exception as e:
            if 'Cannot call "send"' not in str(e):
                logger.error(f"Error sending message to {client_id}: {str(e)}")
            await self.disconnect(client_id)
            return False

# Initialize the WebSocket manager instance
manager = ConnectionManager()

# Define the HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>المساعد الذكي - AraGPT2</title>
    <link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4CAF50;
            --primary-dark: #388E3C;
            --secondary: #2196F3;
            --dark: #2C3E50;
            --light: #F5F7FA;
            --gray: #E0E0E0;
            --dark-gray: #757575;
            --success: #4CAF50;
            --error: #F44336;
        }
        
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Tajawal', sans-serif;
        }
        
        body {
            background-color: #f0f2f5;
            color: #333;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        header {
            text-align: center;
            padding: 20px 0;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        h1 {
            font-size: 2rem;
            margin-bottom: 10px;
        }
        
        .status {
            font-size: 0.9rem;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: #4CAF50;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        #chatbox {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .system-message {
            align-self: center;
            text-align: center;
            max-width: 90%;
        }
        
        .system-message .message-content {
            background-color: #e8f5e9;
            color: #2e7d32;
            font-size: 0.9rem;
            padding: 8px 15px;
            display: inline-block;
        }
        
        .message {
            max-width: 85%;
            margin: 8px 0;
            position: relative;
            animation: fadeIn 0.3s ease-out;
            word-wrap: break-word;
            font-size: 0.95rem;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .message-sender {
            font-weight: 600;
            font-size: 0.85rem;
            margin-bottom: 2px;
            color: #555;
        }
        
        .message-content {
            padding: 10px 15px;
            border-radius: 15px;
            line-height: 1.5;
            position: relative;
        }
        
        .message-time {
            font-size: 0.7rem;
            color: #888;
            margin-top: 2px;
            text-align: right;
        }
        
        .user-message {
            align-self: flex-end;
            align-items: flex-end;
        }
        
        .user-message .message-content {
            background-color: #e3f2fd;
            color: #0d47a1;
            border-bottom-right-radius: 5px;
            border-top-left-radius: 15px;
        }
        
        .bot-message {
            align-self: flex-start;
            align-items: flex-start;
        }
        
        .bot-message .message-content {
            background-color: #f5f5f5;
            color: #212121;
            border-bottom-left-radius: 5px;
            border-top-right-radius: 15px;
        }
        
        .typing-indicator {
            display: inline-flex;
            padding: 10px 15px;
            background-color: #f0f0f0;
            border-radius: 18px;
            margin-bottom: 15px;
            border-bottom-left-radius: 4px;
            align-items: center;
            gap: 5px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background-color: var(--dark-gray);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        .input-container {
            display: flex;
            padding: 15px;
            background: white;
            border-top: 1px solid var(--gray);
            gap: 10px;
        }
        
        #user_input {
            flex: 1;
            padding: 12px 20px;
            border: 2px solid var(--gray);
            border-radius: 25px;
            font-size: 1rem;
            outline: none;
            transition: all 0.3s;
            background-color: #f9f9f9;
        }
        
        #user_input:focus {
            border-color: var(--primary);
            background-color: white;
            box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.2);
        }
        
        #send_button {
            background-color: var(--primary);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0 25px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }
        
        #send_button:hover {
            background-color: var(--primary-dark);
            transform: translateY(-1px);
        }
        
        #send_button:active {
            transform: translateY(0);
        }
        
        #send_button:disabled {
            background-color: var(--gray);
            cursor: not-allowed;
            transform: none;
        }
        
        #send_button svg {
            width: 20px;
            height: 20px;
            fill: currentColor;
        }
        
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            padding: 0 15px 15px;
            background: white;
        }
        
        .suggestion {
            background-color: #e8f5e9;
            color: var(--primary-dark);
            border: 1px solid #c8e6c9;
            border-radius: 15px;
            padding: 6px 12px;
            font-size: 0.85rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .suggestion:hover {
            background-color: #c8e6c9;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes bounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-5px); }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .message {
                max-width: 90%;
            }
            
            #user_input {
                padding: 10px 15px;
            }
            
            #send_button {
                padding: 0 20px;
            }
        }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 10px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>المساعد الذكي</h1>
            <div class="status">
                <span id="statusText">متصل</span>
                <span class="status-dot" id="statusDot"></span>
            </div>
        </header>
        
        <div class="chat-container">
            <div id="chatbox">
                <div class="message bot-message">
                    <div class="message-sender">المساعد الذكي</div>
                    <div class="message-content">مرحباً! أنا مساعدك الذكي. كيف يمكنني مساعدتك اليوم؟</div>
                    <div class="message-time">الآن</div>
                </div>
            </div>
            
            <div class="suggestions" id="suggestions">
                <div class="suggestion" onclick="sendSuggestion(this)">ما هو اسمك؟</div>
                <div class="suggestion" onclick="sendSuggestion(this)">كيف حالك؟</div>
                <div class="suggestion" onclick="sendSuggestion(this)">ما هي آخر الأخبار؟</div>
                <div class="suggestion" onclick="sendSuggestion(this)">ما هو الطقس اليوم؟</div>
            </div>
            
            <div class="input-container">
                <input 
                    type="text" 
                    id="user_input" 
                    placeholder="اكتب رسالتك هنا..." 
                    autocomplete="off"
                    onkeypress="handleKeyPress(event)"
                />
                <button id="send_button" onclick="sendMessage()">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                    </svg>
                    إرسال
                </button>
            </div>
        </div>
    </div>

    <script>
        // DOM Elements
        const chatbox = document.getElementById('chatbox');
        const userInput = document.getElementById('user_input');
        const sendButton = document.getElementById('send_button');
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const suggestions = document.getElementById('suggestions');
        
        // WebSocket connection
        let socket = null;
        let isTyping = false;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        let reconnectDelay = 1000;

        // Initialize WebSocket connection
        function initWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            try {
                socket = new WebSocket(wsUrl);
                updateConnectionStatus('connecting');
                
                socket.onopen = () => {
                    console.log('Connected to WebSocket server');
                    updateConnectionStatus('connected');
                    reconnectAttempts = 0;
                    
                    // Generate a unique client ID if not exists
                    if (!localStorage.getItem('clientId')) {
                        localStorage.setItem('clientId', generateId());
                    }
                    
                    // Send initial connection message
                    sendInitialConnection();
                };

                socket.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'status') {
                            updateConnectionStatus(data.status);
                        } else if (data.type === 'ping') {
                            // Handle ping
                            socket.send(JSON.stringify({ type: 'pong' }));
                        } else {
                            hideTyping();
                            addMessage('المساعد', data.message, 'bot');
                        }
                    } catch (e) {
                        console.error('Error parsing message:', e);
                    }
                };

                socket.onclose = (event) => {
                    console.log('WebSocket connection closed', event);
                    updateConnectionStatus('disconnected');
                    handleReconnect();
                };

                socket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    updateConnectionStatus('error');
                };
                
            } catch (error) {
                console.error('WebSocket initialization error:', error);
                updateConnectionStatus('error');
                handleReconnect();
            }
        }

        // Send initial connection message
        function sendInitialConnection() {
            if (socket && socket.readyState === WebSocket.OPEN) {
                const clientId = localStorage.getItem('clientId') || generateId();
                localStorage.setItem('clientId', clientId);
                
                socket.send(JSON.stringify({ 
                    type: 'connection',
                    clientId: clientId
                }));
            }
        }

        // Handle reconnection
        function handleReconnect() {
            if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                reconnectAttempts++;
                const delay = RECONNECT_DELAY * Math.pow(1.5, reconnectAttempts - 1);
                
                updateConnectionStatus('reconnecting', `إعادة الاتصال... (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);
                
                setTimeout(() => {
                    console.log(`Reconnection attempt ${reconnectAttempts}`);
                    initWebSocket();
                }, delay);
            } else {
                updateConnectionStatus('failed', 'فشل الاتصال. يرجى تحديث الصفحة.');
            }
        }

        // Update connection status UI
        function updateConnectionStatus(status, customMessage = null) {
            const statusMap = {
                'connecting': { text: 'جاري الاتصال...', color: '#FFA000' },
                'connected': { text: 'متصل', color: '#4CAF50' },
                'disconnected': { text: 'غير متصل', color: '#F44336' },
                'reconnecting': { text: customMessage || 'إعادة الاتصال...', color: '#FFA000' },
                'error': { text: 'خطأ في الاتصال', color: '#F44336' },
                'failed': { text: customMessage || 'فشل الاتصال', color: '#F44336' }
            };
            
            const statusInfo = statusMap[status] || statusMap['error'];
            
            if (statusDot) statusDot.style.backgroundColor = statusInfo.color;
            if (statusText) statusText.textContent = statusInfo.text;
            
            // Disable input when disconnected
            if (userInput) {
                userInput.disabled = status !== 'connected';
                sendButton.disabled = status !== 'connected';
            }
        }

        // Generate a simple ID for the client
        function generateId() {
            return 'client_' + Math.random().toString(36).substr(2, 9);
        }

        // Format time as HH:MM
        function formatTime(date = new Date()) {
            return date.toLocaleTimeString('ar-EG', { hour: '2-digit', minute: '2-digit', hour12: true });
        }
        
        // Add message to chat with proper formatting
        function addMessage(sender, message, type) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}-message`;
            
            const time = formatTime();
            
            messageDiv.innerHTML = `
                <div class="message-sender">${sender}</div>
                <div class="message-content">${message}</div>
                <div class="message-time">${time}</div>
            `;
            
            chatbox.appendChild(messageDiv);
            chatbox.scrollTop = chatbox.scrollHeight;
            
            // Auto-hide suggestions after first message
            if (suggestions && type === 'user') {
                suggestions.style.display = 'none';
            }
        }

        // Show typing indicator
        function showTyping() {
            if (isTyping) return;
            
            isTyping = true;
            const typingDiv = document.createElement('div');
            typingDiv.className = 'typing-indicator';
            typingDiv.id = 'typing';
            typingDiv.innerHTML = `
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            `;
            
            chatbox.appendChild(typingDiv);
            chatbox.scrollTop = chatbox.scrollHeight;
        }

        // Hide typing indicator
        function hideTyping() {
            const typing = document.getElementById('typing');
            if (typing) {
                typing.remove();
            }
            isTyping = false;
        }

        // Send message
        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message || !socket || socket.readyState !== WebSocket.OPEN) return;

            // Add user message to chat
            addMessage('أنت', message, 'user');
            userInput.value = '';
            showTyping();

            try {
                // Get client ID
                const clientId = localStorage.getItem('clientId') || generateId();
                localStorage.setItem('clientId', clientId);
                
                // Send message to server
                socket.send(JSON.stringify({ 
                    type: 'message',
                    clientId: clientId,
                    message: message 
                }));
                
            } catch (error) {
                console.error('Error sending message:', error);
                hideTyping();
                addMessage('النظام', 'عذراً، حدث خطأ في إرسال الرسالة', 'system');
            }
        }

        // Handle suggestion click
        function sendSuggestion(element) {
            userInput.value = element.textContent;
            sendMessage();
        }

        // Handle Enter key press
        function handleKeyPress(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        // Auto-resize textarea
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });

        // Initialize on page load
        window.onload = () => {
            initWebSocket();
            userInput.focus();
        };

        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible' && 
                (!socket || socket.readyState !== WebSocket.OPEN)) {
                initWebSocket();
            }
        });
    </script>
</body>
</html>
"""

# Define the routes
@app.get("/", response_class=HTMLResponse)
async def get():
    return HTMLResponse(HTML_TEMPLATE)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"client_{int(time.time())}"
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=manager.timeout
                )
                
                # Handle ping/pong
                try:
                    data_json = json.loads(data)
                    if data_json.get('type') == 'pong':
                        if client_id in manager.active_connections:
                            manager.active_connections[client_id]['last_ping'] = time.time()
                        continue
                    
                    # Handle regular messages
                    if data_json.get('type') == 'message':
                        message = data_json.get('message', '')
                        if message:
                            # Get response from chatbot
                            response = await chatbot.get_response(client_id, message)
                            
                            # Send the response
                            await manager.send_message(response, client_id)
                            
                except json.JSONDecodeError:
                    await manager.send_message("خطأ: تنسيق الرسالة غير صالح", client_id)
                    
            except asyncio.TimeoutError:
                # Check if client is still active
                if client_id in manager.active_connections:
                    last_ping = manager.active_connections[client_id].get('last_ping', 0)
                    if time.time() - last_ping > manager.timeout * 2:
                        logger.warning(f"Client {client_id} timed out")
                        break
                
            except WebSocketDisconnect as e:
                logger.info(f"Client {client_id} disconnected: {e.code}")
                break
                
            except Exception as e:
                logger.error(f"Error in WebSocket for {client_id}: {str(e)}")
                logger.error(traceback.format_exc())
                await manager.send_message("عذراً، حدث خطأ في معالجة طلبك", client_id)
                
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket for {client_id}: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        await manager.disconnect(client_id)

# Health check endpoint
@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "model": MODEL_NAME,
            "device": DEVICE,
            "farasa_available": FARASA_AVAILABLE,
            "connections": len(manager.active_connections)
        }
    )

# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    # Run the server with optimized settings
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable auto-reload to prevent multiple model loading
        log_level="info",
        workers=1,     # Use only 1 worker to avoid memory issues
        limit_concurrency=50,
        timeout_keep_alive=30
    )