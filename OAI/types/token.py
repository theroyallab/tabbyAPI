from pydantic import BaseModel
from typing import List

class CommonTokenRequest(BaseModel):
    add_bos_token: bool = True
    encode_special_tokens: bool = True
    decode_special_tokens: bool = True

    def get_params(self):
        return {
            "add_bos_token": self.add_bos_token,
            "encode_special_tokens": self.encode_special_tokens,
            "decode_special_tokens": self.decode_special_tokens
        }

class TokenEncodeRequest(CommonTokenRequest):
    text: str

class TokenEncodeResponse(BaseModel):
    tokens: List[int]
    length: int

class TokenDecodeRequest(CommonTokenRequest):
    tokens: List[int]

class TokenDecodeResponse(BaseModel):
    text: str

class TokenCountResponse(BaseModel):
    length: int 
