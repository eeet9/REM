from typing import List, Dict, Optional

from transformers import PreTrainedTokenizer, PreTrainedModel


class RoleModel:
    """A class for managing conversation roles and generating responses using a language model."""

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            temperature: float = 0.7,
            max_tokens: int = 4096,
            top_p: float = 0.9,
            top_k: int = 40
    ):
        """
        Initialize the RoleModel.

        Args:
            model: Pretrained language model
            tokenizer: Tokenizer for the model
            temperature: Sampling temperature for generation
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.ask_history: List[Dict[str, str]] = []  # History of ask messages
        self.answer_history: List[Dict[str, str]] = []  # History of answer messages

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def add_content(
            self,
            content: str,
            role: str,
            role_type: str,
            metadata: Optional[Dict] = None
    ) -> None:
        """
        Add content to conversation history.

        Args:
            content: The text content to add
            role: Either "ask" or "answer"
            role_type: The role type (e.g., "user", "assistant")
            metadata: Optional additional metadata
        """
        entry = {"role": role_type, "content": content}
        if metadata:
            entry.update(metadata)

        if role == "ask":
            self.ask_history.append(entry)
        elif role == "answer":
            self.answer_history.append(entry)

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.ask_history.clear()
        self.answer_history.clear()

    def _generate_response(
            self,
            history: List[Dict[str, str]],
            max_new_tokens: int = 1024
    ) -> str:
        """
        Internal method to generate responses from history.

        Args:
            history: Conversation history to use as context
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Generated response text
        """
        if not history:
            return ""

        try:
            # Prepare input
            text = self.tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)

            # Generate response
            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                top_p=self.top_p,
                temperature=self.temperature,
                top_k=self.top_k,
                pad_token_id=pad_token_id,
                do_sample=True
            )

            # Decode response
            response = self.tokenizer.decode(
                generated_ids[0][len(model_inputs.input_ids[0]):],
                skip_special_tokens=True
            )
            return response.strip()

        except Exception as e:
            print(f"Generation error: {str(e)}")
            return ""

    def run_ask(self, max_new_tokens: int = 1024) -> str:
        """
        Generate a response based on ask history.

        Args:
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Generated response text
        """
        return self._generate_response(self.ask_history, max_new_tokens)

    def run_answer(self, max_new_tokens: int = 1024) -> str:
        """
        Generate a response based on answer history.

        Args:
            max_new_tokens: Maximum new tokens to generate

        Returns:
            Generated response text
        """
        return self._generate_response(self.answer_history, max_new_tokens)

    def get_history(self, role: str) -> List[Dict[str, str]]:
        """
        Get conversation history for a specific role.

        Args:
            role: Either "ask" or "answer"

        Returns:
            The requested history
        """
        if role == "ask":
            return self.ask_history
        elif role == "answer":
            return self.answer_history
        return []