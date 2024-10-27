from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from groq import Groq
import time
import sys
import os
from typing import Optional, Dict, List

def typrint(text):
    """Typewriter effect for printing"""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)

class groqchain:
    # Default prioritized list of models for fallback
    MODEL_PRIORITY = [
        "mixtral-8x7b-32768",      # Primary: 32K context, best quality
        "llama3-8b-8192",          # Fallback 1: 8K context, fast, high TPM
        "llama-3.1-8b-instant"     # Fallback 2: 131K context, reliable
    ]

    # Best models for code generation
    CODE_MODEL=["llama-3.2-11b-text-preview", "llama3-groq-70b-8192-tool-use-preview"]

    def __init__(self, 
                 mem: str = 'n', 
                 system_message: Optional[str] = None, 
                 temperature: float = 0.1,
                 verbose: bool = True,
                 model: Optional[str] = None):
        """
        Initialize the GroqChain instance.
        
        Args:
            mem: Enable memory ('y'/'n')
            system_message: System prompt for the conversation
            temperature: Model temperature (0-1)
            verbose: Enable detailed logging
            model: Optional specific model to start with
        """
        if mem not in ('y', 'n'):
            raise ValueError("mem must be 'y' or 'n'")
        
        load_dotenv()
        self.system_message = system_message
        self.memory = ConversationBufferMemory() if mem == 'y' else None
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.temperature = temperature
        self.verbose = verbose
        
        # Create custom model priority list if user specifies a model
        if model:
            if model in self.MODEL_PRIORITY:
                # If user's model is in default priority list, start from there
                self.current_model_index = self.MODEL_PRIORITY.index(model)
            else:
                # If user's model is different, put it first and append others
                self.MODEL_PRIORITY = [model] + [m for m in self.MODEL_PRIORITY if m != model]
                self.current_model_index = 0
        else:
            self.current_model_index = 0
        
        if self.verbose:
            print(f"Initialized with model: {self.current_model}")
            print(f"Memory enabled: {self.memory is not None}")
            if system_message:
                print(f"System message: {system_message[:100]}...")
    
    @property
    def current_model(self) -> str:
        """Get the currently active model"""
        return self.MODEL_PRIORITY[self.current_model_index]
    
    def switch_to_next_model(self) -> bool:
        """
        Switch to the next model in priority list.
        Returns:
            bool: True if switched successfully, False if no models left
        """
        if self.current_model_index < len(self.MODEL_PRIORITY) - 1:
            prev_model = self.current_model
            self.current_model_index += 1
            if self.verbose:
                print(f"\nSwitching from {prev_model} to {self.current_model}")
                if self.memory:
                    print(f"Preserving conversation history ({len(self.memory.chat_memory.messages)} messages)")
            return True
        return False
    
    def reset_to_primary_model(self):
        """Reset to use the primary model"""
        self.current_model_index = 0
        if self.verbose:
            print(f"Reset to primary model: {self.current_model}")
    
    def set_model(self, model: str):
        """
        Set a specific model and reorganize priority list
        Args:
            model: Model identifier to set as primary
        """
        if model in self.MODEL_PRIORITY:
            self.current_model_index = self.MODEL_PRIORITY.index(model)
        else:
            self.MODEL_PRIORITY = [model] + [m for m in self.MODEL_PRIORITY if m != model]
            self.current_model_index = 0
            
        if self.verbose:
            print(f"Switched to model: {self.current_model}")
            print(f"Updated fallback order: {' -> '.join(self.MODEL_PRIORITY)}")

    # [Rest of the methods remain the same as in your provided code]
    def add_user_message(self, msg: str):
        """Add a user message to memory"""
        if self.memory:
            self.memory.chat_memory.add_user_message(msg)

    def add_ai_message(self, msg: str):
        """Add an AI message to memory"""
        if self.memory:
            self.memory.chat_memory.add_ai_message(msg)
    
    def get_response(self, query: str, typewriter: bool = False) -> str:
        """
        Get a response from the model with automatic fallback handling.
        
        Args:
            query: The user's input query
            typewriter: Enable typewriter effect for response
        
        Returns:
            str: The model's response
            
        Raises:
            Exception: If all models fail or other errors occur
        """
        if self.verbose:
            print(f"\nProcessing query with {self.current_model}")
        
        messages = []
        
        # Add system message if present
        if self.system_message:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        
        # Add chat history if memory is enabled
        history_messages = 0
        if self.memory:
            chat_history = self.get_chat_history()
            if chat_history and "No memories available" not in chat_history:
                for message in self.memory.chat_memory.messages:
                    role = "user" if "Human" in str(message) else "assistant"
                    content = str(message).split("content=")[1].strip("')")
                    messages.append({
                        "role": role,
                        "content": content
                    })
                    history_messages += 1
        
        if self.verbose and history_messages > 0:
            print(f"Including {history_messages} messages from conversation history")
        
        # Add current query
        messages.append({
            "role": "user",
            "content": query
        })
        
        while True:
            try:
                if self.verbose:
                    print(f"Attempting inference with {self.current_model}...")
                
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=self.current_model,
                    temperature=self.temperature
                )
                
                response = chat_completion.choices[0].message.content
                
                # Add to memory if enabled
                if self.memory:
                    self.add_user_message(query)
                    self.add_ai_message(response)
                    if self.verbose:
                        print(f"Updated conversation history (now {len(self.memory.chat_memory.messages)} messages)")
                
                if self.verbose:
                    print(f"Successfully generated response with {self.current_model}")
                
                # Handle response output
                if typewriter:
                    typrint(response)
                    print("\n")
                    return response
                else:
                    return response
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for specific error types
                is_context_error = "context" in error_str and "length" in error_str
                is_token_limit_error = any(phrase in error_str for phrase 
                                         in ["token", "rate limit", "quota"])
                
                if is_context_error or is_token_limit_error:
                    if self.verbose:
                        print(f"\nError with {self.current_model}: {str(e)}")
                    if not self.switch_to_next_model():
                        raise Exception("All models exhausted. Last error: " + str(e))
                    if self.verbose:
                        print("Retrying same query with new model...")
                    continue
                else:
                    raise e

    def get_chat_history(self) -> str:
        """Get the current chat history"""
        return str(self.memory.chat_memory) if self.memory else "No memories available. Input mem='y' in the Groqchain object to save memories"

    def set_system_message(self, system_message: str):
        """Set or update the system message"""
        self.system_message = system_message
        if self.verbose:
            print(f"Updated system message: {system_message[:100]}...")
    
    def get_system_message(self) -> Optional[str]:
        """Get the current system message"""
        return self.system_message
    
    def get_last_ai_msg(self) -> Optional[str]:
        """Get the last AI message from chat history"""
        given_string = self.get_chat_history()
        aimessage_start = given_string.find('AIMessage(content=')
        aimessage_end = given_string.find(')]', aimessage_start)

        if aimessage_start != -1 and aimessage_end != -1:
            aimessage_content = given_string[aimessage_start + len('AIMessage(content=') + 1: aimessage_end - 1]
            return aimessage_content
        else:
            if self.verbose:
                print("No AIMessage content found.")
            return None