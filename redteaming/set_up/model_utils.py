from openai import OpenAI, AsyncOpenAI
import os
import asyncio
import re
from typing import Tuple, Optional, Dict, List, Any, Union
import requests
import json
import math


class Model():
    def __init__(self, api_key = os.getenv("API_KEY"), base_url = "https://api.fireworks.ai/inference/v1/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def get_completion(self, prompt = None, model = "accounts/fireworks/models/gpt-oss-20b", temperature: float = 0, top_p: float = 1, reasoning_level: str = "low", system_message: str = None, input_data = None, developer_message: str = None, max_tokens: int = 4096) -> Tuple[str, str, str, Dict[str, Any]]:
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})

        if developer_message:
            messages.append({"role": "developer", "content": developer_message})
        
        messages.append({"role": "user", "content": prompt})

        # input data supercedes all other messages
        if input_data:
            messages = input_data
        
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "n": 1,
            "top_p": top_p,
            "top_k": 40,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "temperature": temperature,
            "messages": messages,
            "raw_output": True,
            "reasoning_effort": reasoning_level,
            "echo": True,
            "logprobs": 1 # checking for logprobs, can be 1 <= x <= 5 (or null if not needed)
        }
        
        try:
            response = requests.request("POST", self.url, headers=self.headers, data=json.dumps(payload))
            return self._parse_output(response.json())
            # return response.json()
        except Exception as e:
            print(f"Error: {e}")
            return f"Error {e}", f"Error {e}", f"Error {e}", f"Error {e}"
    
    def _parse_output(self, output: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """
        Parse the raw output from FireworksModel.get_completion() and extract key components.
        
        Args:
            output: The raw output dictionary from the API response
            
        Returns:
            A tuple containing:
            1. content: str
            2. reasoning_content: str
            3. completion: str
            4. usage: dict
        """
        try:
            content = None
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                if 'raw_output' in choice:
                    completion = choice['raw_output']['completion']
                    content = completion.split("<|start|>assistant<|channel|>final<|message|>")[-1]
            
            reasoning_content = None
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                if 'message' in choice and 'reasoning_content' in choice['message']:
                    reasoning_content = choice['message']['reasoning_content']
            

            completion = None
            if 'choices' in output and len(output['choices']) > 0:
                choice = output['choices'][0]
                if 'raw_output' in choice:
                    completion = choice['raw_output']['completion']
            
            logprobs_dict = output['choices'][0]['logprobs']
            logprobs = logprobs_dict.get('token_logprobs', None)
            tokens = logprobs_dict.get('tokens', None)
            confidence = self._calculate_sequence_confidence(logprobs, tokens)
            
            return content, reasoning_content, completion, confidence
            
        except Exception as e:
            try:
                logprobs_dict = output['raw_output']['completion_logprobs']
            except:
                print(f"Output: {output}")
            tokens_dicts = logprobs_dict['content']
            logprobs = []
            tokens = []
            for token in tokens_dicts:
                logprobs.append(token['logprob'])
                tokens.append(token['token'])
            confidence = self._calculate_sequence_confidence_extension(logprobs, tokens)
            return "parsing failed", "parsing failed", "parsing failed", confidence

    def _calculate_token_confidence(self, logprobs: List[float], tokens: List[str]) -> float:
        """
        Calculate the confidence of the model's response based on the logprobs.
        """

        # converting logprobs to probabilities 
        probabilities = [math.exp(logprob) for logprob in logprobs]

        # zipping the logprobs and tokens
        logprobs_tokens = list(zip(logprobs, tokens))

        return logprobs_tokens
    
    def _calculate_sequence_confidence(self, logprobs: List[float], tokens: List[str]) -> float:
        """
        Calculate the confidence of the model's response based on the logprobs.
        """
        # finding the index of the first response token
        first_response_token_index = len(tokens) - 1 - tokens[::-1].index('<|message|>') + 1

        # Only taking the logprobs of the response tokens
        logprobs = logprobs[first_response_token_index:]

        # summing the logprobs
        confidence = sum(logprobs)

        return confidence
    
    def _calculate_sequence_confidence_extension(self, logprobs: List[float], tokens: List[str]) -> float:
        """
        Calculate the confidence of the model's response based on the logprobs.
        """
        # finding the index of the first response token
        first_response_token_index = len(tokens) - 1 - tokens[::-1].index('<|message|>') + 1

        # finding the index of the end token
        try:
            end_token_index = tokens.index('<|end|>', first_response_token_index)
        except ValueError:
            # if <|end|> token is not found, use all tokens from the response start
            end_token_index = len(tokens)

        # Only taking the logprobs of the response tokens up to the end token
        logprobs = logprobs[first_response_token_index:end_token_index]

        # summing the logprobs
        confidence = sum(logprobs)

        return confidence

class MultipleChoiceTest(Model):
    def __init__(self, api_key = os.getenv("API_KEY"), base_url = "https://api.fireworks.ai/inference/v1/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url
        self.url = "https://api.fireworks.ai/inference/v1/chat/completions"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def loglikelihood(self, requests) -> list:
        ''' Calculates the loglikelihood of the model's response to the request for MMLU and HellaSwag Tests'''
        results = []
        for request in requests:
            prompt = request[0]
            response = request[1]
            input_data = []
            input_data.append({"role": "user", "content": prompt})
            input_data.append({"role": "assistant", "content": response})
            _,_,_,confidence = self.get_completion(input_data=input_data, max_tokens=0)
            results.append((confidence,False))
        return results  

    