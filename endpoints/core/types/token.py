"""Tokenization types"""

from pydantic import BaseModel, Field
from typing import Dict, List, Union


class CommonTokenRequest(BaseModel):
    """Represents a common tokenization request."""

    add_bos_token: bool = Field(
        True, description="Add the BOS (beginning of sequence) token"
    )
    encode_special_tokens: bool = Field(True, description="Encode special tokens")
    decode_special_tokens: bool = Field(True, description="Decode special tokens")

    def get_params(self):
        """Get the parameters for tokenization."""
        return {
            "add_bos_token": self.add_bos_token,
            "encode_special_tokens": self.encode_special_tokens,
            "decode_special_tokens": self.decode_special_tokens,
        }


class TokenEncodeRequest(CommonTokenRequest):
    """Represents a tokenization request."""

    text: Union[str, List[Dict[str, str]]] = Field(description="The string to encode")


class TokenEncodeResponse(BaseModel):
    """Represents a tokenization response."""

    tokens: List[int] = Field(description="The tokens")
    length: int = Field(description="The length of the tokens")


class TokenDecodeRequest(CommonTokenRequest):
    """ " Represents a detokenization request."""

    tokens: List[int] = Field(description="The string to encode")


class TokenDecodeResponse(BaseModel):
    """Represents a detokenization response."""

    text: str = Field(description="The decoded text")


class TokenCountResponse(BaseModel):
    """Represents a token count response."""

    length: int = Field(description="The length of the text")
