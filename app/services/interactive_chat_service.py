# app/services/interactive_chat_service.py
import logging
import json
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class ConversationIntent(Enum):
    """Detected user intents"""
    GREETING = "greeting"
    CREATE_REST_CONNECTION = "create_rest_connection"
    CREATE_DATABASE_CONNECTION = "create_database_connection"
    LIST_CONNECTIONS = "list_connections"
    HELP = "help"
    QUERY_INFO = "query_info"
    UNKNOWN = "unknown"


class ConversationStep(Enum):
    """Steps in connection creation workflow"""
    INITIAL = "initial"
    COLLECT_ENDPOINT = "collect_endpoint"
    COLLECT_AUTH_METHOD = "collect_auth_method"
    COLLECT_CREDENTIALS = "collect_credentials"
    COLLECT_TIMEOUT = "collect_timeout"
    CONFIRM = "confirm"
    COMPLETE = "complete"


@dataclass
class Message:
    """Single message in conversation"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    
    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ConversationState:
    """State of an ongoing conversation"""
    session_id: str
    intent: ConversationIntent
    step: ConversationStep
    collected_data: Dict[str, Any]
    required_fields: List[str]
    conversation_history: List[Message]
    created_at: datetime
    last_activity: datetime
    
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'intent': self.intent.value,
            'step': self.step.value,
            'collected_data': self.collected_data,
            'required_fields': self.required_fields,
            'conversation_history': [msg.to_dict() for msg in self.conversation_history],
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat()
        }


class InteractiveChatService:
    """
    Interactive chat service for conversational AI
    Uses the same LLM instance as JSON mapping but with different prompting strategy
    """
    
    def __init__(self, llama_manager):
        """
        Initialize chat service with shared LLM manager
        
        Args:
            llama_manager: Shared LlamaManager instance
        """
        self.llama_manager = llama_manager
        
        # In-memory session storage (use Redis in production)
        self.sessions: Dict[str, ConversationState] = {}
        
        # Session timeout (30 minutes of inactivity)
        self.session_timeout = timedelta(minutes=30)
        
        # Connection type configurations
        self.connection_configs = self._load_connection_configs()
        
        logger.info("✅ InteractiveChatService initialized with shared LLM")
    
    def _load_connection_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load connection type configurations"""
        return {
            "REST": {
                "required_fields": ["endpoint_url", "auth_method"],
                "optional_fields": ["timeout", "retry_count", "headers"],
                "auth_methods": ["None", "Basic", "Bearer Token", "OAuth2", "API Key"],
                "workflow_steps": [
                    ConversationStep.COLLECT_ENDPOINT,
                    ConversationStep.COLLECT_AUTH_METHOD,
                    ConversationStep.COLLECT_CREDENTIALS,
                    ConversationStep.COLLECT_TIMEOUT,
                    ConversationStep.CONFIRM
                ]
            },
            "DATABASE": {
                "required_fields": ["db_type", "host", "port", "database_name", "username", "password"],
                "optional_fields": ["connection_pool_size", "timeout"],
                "db_types": ["MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQL Server"],
                "workflow_steps": [
                    ConversationStep.COLLECT_ENDPOINT,  # Reusing for host/port
                    ConversationStep.COLLECT_CREDENTIALS,
                    ConversationStep.COLLECT_TIMEOUT,
                    ConversationStep.CONFIRM
                ]
            }
        }
    
    def process_message(self, session_id: str, user_message: str) -> Dict[str, Any]:
        """
        Main entry point: Process user message and return response
        
        Args:
            session_id: Unique session identifier
            user_message: User's message text
            
        Returns:
            Response dictionary with assistant message and metadata
        """
        try:
            start_time = time.time()
            
            # Clean session storage (remove expired sessions)
            self._cleanup_expired_sessions()
            
            # Get or create session
            session = self._get_or_create_session(session_id, user_message)
            
            # Add user message to history
            self._add_message(session, "user", user_message)
            
            # Detect intent if not already set
            if session.intent == ConversationIntent.UNKNOWN:
                session.intent = self._detect_intent(user_message)
                logger.info(f"Detected intent: {session.intent.value}")
            
            # Generate response based on intent and current step
            assistant_response = self._generate_response(session, user_message)
            
            # Add assistant response to history
            self._add_message(session, "assistant", assistant_response)
            
            # Update session state
            session.last_activity = datetime.now()
            self.sessions[session_id] = session
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "session_id": session_id,
                "response": assistant_response,
                "intent": session.intent.value,
                "step": session.step.value,
                "collected_data": session.collected_data,
                "conversation_complete": session.step == ConversationStep.COMPLETE,
                "processing_time": f"{processing_time:.2f}s",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_or_create_session(self, session_id: str, first_message: str = None) -> ConversationState:
        """Get existing session or create new one"""
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Check if session expired
            if datetime.now() - session.last_activity > self.session_timeout:
                logger.info(f"Session {session_id} expired, creating new one")
                return self._create_new_session(session_id, first_message)
            
            return session
        else:
            return self._create_new_session(session_id, first_message)
    
    def _create_new_session(self, session_id: str, first_message: str = None) -> ConversationState:
        """Create new conversation session"""
        
        now = datetime.now()
        
        session = ConversationState(
            session_id=session_id,
            intent=ConversationIntent.UNKNOWN,
            step=ConversationStep.INITIAL,
            collected_data={},
            required_fields=[],
            conversation_history=[],
            created_at=now,
            last_activity=now
        )
        
        # Add initial greeting if it's a brand new conversation
        if not first_message or self._is_greeting(first_message):
            greeting = "Hello! I'm your technical assistant. I can help you with:\n\n" \
                      "• Creating REST connections\n" \
                      "• Creating database connections\n" \
                      "• Listing existing connections\n" \
                      "• General information and help\n\n" \
                      "How can I assist you today?"
            
            self._add_message(session, "assistant", greeting)
        
        self.sessions[session_id] = session
        logger.info(f"Created new session: {session_id}")
        
        return session
    
    def _detect_intent(self, message: str) -> ConversationIntent:
        """Detect user intent from message"""
        
        message_lower = message.lower()
        
        # Greeting patterns
        if self._is_greeting(message):
            return ConversationIntent.GREETING
        
        # Create REST connection
        if any(keyword in message_lower for keyword in ['create rest', 'rest connection', 'new rest', 'add rest', 'setup rest']):
            return ConversationIntent.CREATE_REST_CONNECTION
        
        # Create database connection
        if any(keyword in message_lower for keyword in ['create database', 'database connection', 'new database', 'db connection', 'connect database']):
            return ConversationIntent.CREATE_DATABASE_CONNECTION
        
        # List connections
        if any(keyword in message_lower for keyword in ['list connection', 'show connection', 'my connection', 'existing connection', 'what connection']):
            return ConversationIntent.LIST_CONNECTIONS
        
        # Help
        if any(keyword in message_lower for keyword in ['help', 'what can you do', 'how to', 'assist']):
            return ConversationIntent.HELP
        
        # Query information
        if any(keyword in message_lower for keyword in ['what is', 'explain', 'tell me about', 'how does']):
            return ConversationIntent.QUERY_INFO
        
        return ConversationIntent.UNKNOWN
    
    def _is_greeting(self, message: str) -> bool:
        """Check if message is a greeting"""
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings']
        return any(greeting in message.lower() for greeting in greetings)
    
    def _generate_response(self, session: ConversationState, user_message: str) -> str:
        """Generate contextual response based on intent and conversation state"""
        
        # Handle different intents
        if session.intent == ConversationIntent.GREETING:
            return self._handle_greeting(session)
        
        elif session.intent == ConversationIntent.CREATE_REST_CONNECTION:
            return self._handle_rest_connection_creation(session, user_message)
        
        elif session.intent == ConversationIntent.CREATE_DATABASE_CONNECTION:
            return self._handle_database_connection_creation(session, user_message)
        
        elif session.intent == ConversationIntent.LIST_CONNECTIONS:
            return self._handle_list_connections(session)
        
        elif session.intent == ConversationIntent.HELP:
            return self._handle_help(session)
        
        elif session.intent == ConversationIntent.QUERY_INFO:
            return self._handle_query_info(session, user_message)
        
        else:
            # Unknown intent - use LLM for general response
            return self._generate_llm_response(session, user_message)
    
    def _handle_greeting(self, session: ConversationState) -> str:
        """Handle greeting intent"""
        session.step = ConversationStep.INITIAL
        return "Hello! I'm here to help you with your system configurations. What would you like to do today?"
    
    def _handle_rest_connection_creation(self, session: ConversationState, user_message: str) -> str:
        """Handle REST connection creation workflow"""
        
        config = self.connection_configs["REST"]
        
        # Initialize workflow if just starting
        if session.step == ConversationStep.INITIAL:
            session.step = ConversationStep.COLLECT_ENDPOINT
            session.required_fields = config["required_fields"].copy()
            session.collected_data["connection_type"] = "REST"
            return "Great! Let's create a REST connection. First, what is the endpoint URL? (e.g., https://api.example.com)"
        
        # Step 1: Collect endpoint URL
        elif session.step == ConversationStep.COLLECT_ENDPOINT:
            endpoint = self._extract_url(user_message)
            if endpoint:
                session.collected_data["endpoint_url"] = endpoint
                session.step = ConversationStep.COLLECT_AUTH_METHOD
                
                auth_methods = "\n".join([f"{i+1}. {method}" for i, method in enumerate(config["auth_methods"])])
                return f"Perfect! Endpoint set to: {endpoint}\n\n" \
                       f"Now, which authentication method would you like to use?\n\n{auth_methods}\n\n" \
                       f"Please enter the number or name of your choice."
            else:
                return "I couldn't detect a valid URL. Please provide the endpoint URL (e.g., https://api.example.com)"
        
        # Step 2: Collect authentication method
        elif session.step == ConversationStep.COLLECT_AUTH_METHOD:
            auth_method = self._extract_auth_method(user_message, config["auth_methods"])
            if auth_method:
                session.collected_data["auth_method"] = auth_method
                
                # If auth method requires credentials, collect them
                if auth_method in ["Basic", "Bearer Token", "OAuth2", "API Key"]:
                    session.step = ConversationStep.COLLECT_CREDENTIALS
                    return self._get_credentials_prompt(auth_method)
                else:
                    # No auth, skip to timeout
                    session.step = ConversationStep.COLLECT_TIMEOUT
                    return "Authentication method set to 'None'.\n\n" \
                           "Would you like to set a timeout value in seconds? (Default is 30 seconds, or type 'skip' to use default)"
            else:
                auth_methods = "\n".join([f"{i+1}. {method}" for i, method in enumerate(config["auth_methods"])])
                return f"I didn't recognize that authentication method. Please choose from:\n\n{auth_methods}"
        
        # Step 3: Collect credentials
        elif session.step == ConversationStep.COLLECT_CREDENTIALS:
            auth_method = session.collected_data.get("auth_method")
            credentials = self._extract_credentials(user_message, auth_method)
            
            if credentials:
                session.collected_data["credentials"] = credentials
                session.step = ConversationStep.COLLECT_TIMEOUT
                return "Credentials saved securely.\n\n" \
                       "Would you like to set a timeout value in seconds? (Default is 30 seconds, or type 'skip' to use default)"
            else:
                return f"I couldn't extract the credentials. {self._get_credentials_prompt(auth_method)}"
        
        # Step 4: Collect timeout
        elif session.step == ConversationStep.COLLECT_TIMEOUT:
            if "skip" in user_message.lower() or "default" in user_message.lower():
                session.collected_data["timeout"] = 30
            else:
                timeout = self._extract_number(user_message)
                if timeout and 1 <= timeout <= 300:
                    session.collected_data["timeout"] = timeout
                else:
                    session.collected_data["timeout"] = 30
            
            session.step = ConversationStep.CONFIRM
            return self._generate_confirmation_summary(session)
        
        # Step 5: Confirm
        elif session.step == ConversationStep.CONFIRM:
            if self._is_confirmation(user_message):
                # Create the connection
                result = self._create_rest_connection(session.collected_data)
                session.step = ConversationStep.COMPLETE
                
                if result["success"]:
                    return f"✅ REST connection created successfully!\n\n" \
                           f"Connection ID: {result['connection_id']}\n" \
                           f"Name: {result['name']}\n" \
                           f"Endpoint: {result['endpoint']}\n\n" \
                           f"You can now use this connection in your workflows. Is there anything else I can help you with?"
                else:
                    return f"❌ Failed to create connection: {result['error']}\n\n" \
                           f"Would you like to try again?"
            else:
                session.step = ConversationStep.INITIAL
                session.collected_data = {}
                return "Connection creation cancelled. Would you like to start over or do something else?"
        
        return "I'm not sure how to proceed. Could you clarify?"
    
    def _handle_database_connection_creation(self, session: ConversationState, user_message: str) -> str:
        """Handle database connection creation workflow"""
        
        config = self.connection_configs["DATABASE"]
        
        if session.step == ConversationStep.INITIAL:
            session.step = ConversationStep.COLLECT_ENDPOINT  # Reusing for DB type selection
            session.required_fields = config["required_fields"].copy()
            session.collected_data["connection_type"] = "DATABASE"
            
            db_types = "\n".join([f"{i+1}. {db}" for i, db in enumerate(config["db_types"])])
            return f"Let's create a database connection! Which database type?\n\n{db_types}\n\nPlease enter the number or name."
        
        # Implementation continues similar to REST connection...
        # For brevity, returning placeholder
        return "Database connection creation is in progress. (Implementation follows REST pattern)"
    
    def _handle_list_connections(self, session: ConversationState) -> str:
        """Handle listing existing connections"""
        
        # Mock data - replace with actual connection retrieval
        connections = [
            {"id": "conn_001", "type": "REST", "name": "Main API", "endpoint": "https://api.example.com"},
            {"id": "conn_002", "type": "DATABASE", "name": "MySQL DB", "host": "localhost:3306"}
        ]
        
        if not connections:
            return "You don't have any connections configured yet. Would you like to create one?"
        
        response = "Here are your existing connections:\n\n"
        for i, conn in enumerate(connections, 1):
            response += f"{i}. **{conn['name']}** ({conn['type']})\n"
            response += f"   ID: {conn['id']}\n"
            if conn['type'] == 'REST':
                response += f"   Endpoint: {conn['endpoint']}\n"
            elif conn['type'] == 'DATABASE':
                response += f"   Host: {conn['host']}\n"
            response += "\n"
        
        response += "Would you like to view details of a specific connection?"
        session.step = ConversationStep.COMPLETE
        
        return response
    
    def _handle_help(self, session: ConversationState) -> str:
        """Handle help request"""
        session.step = ConversationStep.COMPLETE
        
        return """Here's what I can help you with:

**Connection Management:**
• Create REST API connections with various authentication methods
• Create database connections (MySQL, PostgreSQL, MongoDB, etc.)
• List and view existing connections

**How to use me:**
Just tell me what you want in natural language! For example:
• "Create a REST connection"
• "I need to connect to a database"
• "Show me my connections"
• "What authentication methods are supported?"

**Need specific help?**
Just ask me a question about any topic, and I'll do my best to assist!

What would you like to do?"""
    
    def _handle_query_info(self, session: ConversationState, user_message: str) -> str:
        """Handle information query using LLM"""
        session.step = ConversationStep.COMPLETE
        return self._generate_llm_response(session, user_message)
    
    def _generate_llm_response(self, session: ConversationState, user_message: str) -> str:
        """Generate response using LLM for general queries"""
        
        # Build context from conversation history
        conversation_context = self._build_conversation_context(session)
        
        # Create prompt for LLM
        system_prompt = """You are a helpful technical assistant for a system integration platform.
You help users configure connections, understand concepts, and navigate the system.

Keep responses:
- Clear and concise (2-4 sentences max)
- Professional but friendly
- Technically accurate
- Action-oriented when appropriate

Current conversation context:
{context}

User's latest question: {question}

Provide a helpful response:"""
        
        prompt = system_prompt.format(
            context=conversation_context,
            question=user_message
        )
        
        # Format for CodeLlama
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Generate response with chat-optimized parameters
        try:
            response = self.llama_manager.generate_response(
                prompt=formatted_prompt,
                max_tokens=300,  # Shorter for chat
                temperature=0.8,  # More creative for conversation
                timeout=15  # Quick timeout for chat
            )
            
            if response:
                # Clean up response
                response = response.strip()
                # Remove any instruction tokens
                response = response.replace("[/INST]", "").replace("[INST]", "").strip()
                return response
            else:
                return "I apologize, but I'm having trouble generating a response right now. Could you rephrase your question?"
                
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I encountered an error processing your request. Please try again."
    
    def _build_conversation_context(self, session: ConversationState, max_messages: int = 6) -> str:
        """Build conversation context string from recent history"""
        
        # Get last N messages
        recent_messages = session.conversation_history[-max_messages:]
        
        context_lines = []
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            context_lines.append(f"{role}: {msg.content[:100]}")  # Truncate long messages
        
        return "\n".join(context_lines)
    
    def _add_message(self, session: ConversationState, role: str, content: str):
        """Add message to conversation history"""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        session.conversation_history.append(message)
    
    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL from text"""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, text)
        return match.group(0) if match else None
    
    def _extract_auth_method(self, text: str, valid_methods: List[str]) -> Optional[str]:
        """Extract authentication method from text"""
        text_lower = text.lower()
        
        # Check for number selection
        if text.strip().isdigit():
            index = int(text.strip()) - 1
            if 0 <= index < len(valid_methods):
                return valid_methods[index]
        
        # Check for method name
        for method in valid_methods:
            if method.lower() in text_lower:
                return method
        
        return None
    
    def _extract_credentials(self, text: str, auth_method: str) -> Optional[Dict[str, str]]:
        """Extract credentials based on auth method"""
        
        if auth_method == "Bearer Token":
            # Look for token
            token_match = re.search(r'token[:\s]+([A-Za-z0-9\-._~+/]+)', text, re.IGNORECASE)
            if token_match:
                return {"token": token_match.group(1)}
        
        elif auth_method == "API Key":
            # Look for API key
            key_match = re.search(r'(?:key|api[_\s]?key)[:\s]+([A-Za-z0-9\-._~+/]+)', text, re.IGNORECASE)
            if key_match:
                return {"api_key": key_match.group(1)}
        
        elif auth_method == "Basic":
            # Look for username and password
            username_match = re.search(r'username[:\s]+([^\s,]+)', text, re.IGNORECASE)
            password_match = re.search(r'password[:\s]+([^\s,]+)', text, re.IGNORECASE)
            if username_match and password_match:
                return {
                    "username": username_match.group(1),
                    "password": password_match.group(1)
                }
        
        # Fallback: treat entire message as credential value
        return {"value": text.strip()}
    
    def _extract_number(self, text: str) -> Optional[int]:
        """Extract number from text"""
        numbers = re.findall(r'\d+', text)
        return int(numbers[0]) if numbers else None
    
    def _get_credentials_prompt(self, auth_method: str) -> str:
        """Get appropriate prompt for credentials based on auth method"""
        
        prompts = {
            "Bearer Token": "Please provide your Bearer token:",
            "API Key": "Please provide your API key:",
            "Basic": "Please provide your username and password (format: username: xxx, password: yyy):",
            "OAuth2": "Please provide your OAuth2 client ID and client secret:"
        }
        
        return prompts.get(auth_method, "Please provide your credentials:")
    
    def _is_confirmation(self, text: str) -> bool:
        """Check if text is a confirmation"""
        confirmations = ['yes', 'confirm', 'correct', 'proceed', 'ok', 'yeah', 'yep', 'sure']
        return any(word in text.lower() for word in confirmations)
    
    def _generate_confirmation_summary(self, session: ConversationState) -> str:
        """Generate confirmation summary"""
        
        data = session.collected_data
        
        summary = "Perfect! Here's a summary of your REST connection:\n\n"
        summary += f"• Endpoint: {data.get('endpoint_url', 'N/A')}\n"
        summary += f"• Authentication: {data.get('auth_method', 'N/A')}\n"
        summary += f"• Timeout: {data.get('timeout', 30)} seconds\n"
        
        if data.get('credentials'):
            summary += f"• Credentials: Configured ✓\n"
        
        summary += "\nWould you like to proceed with creating this connection? (yes/no)"
        
        return summary
    
    def _create_rest_connection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create REST connection (mock implementation)"""
        
        # In production, this would:
        # 1. Validate all data
        # 2. Store in database
        # 3. Test connection
        # 4. Return connection object
        
        import uuid
        
        connection_id = f"rest_{uuid.uuid4().hex[:8]}"
        
        # Mock successful creation
        return {
            "success": True,
            "connection_id": connection_id,
            "name": f"REST Connection {connection_id}",
            "endpoint": data.get("endpoint_url"),
            "auth_method": data.get("auth_method"),
            "created_at": datetime.now().isoformat()
        }
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions from memory"""
        
        now = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if now - session.last_activity > self.session_timeout:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")
    
    def get_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of a session"""
        
        if session_id in self.sessions:
            return self.sessions[session_id].to_dict()
        return None
    
    def clear_session(self, session_id: str) -> bool:
        """Clear/delete a session"""
        
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")
            return True
        return False
    
    def get_active_sessions_count(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)