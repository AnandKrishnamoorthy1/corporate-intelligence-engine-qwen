"""
Qwen LLM Integration Module

Provides wrapper functions for interacting with Alibaba's Qwen models
through the OpenAI-compatible DashScope API (QwenCloud endpoint).

Uses OpenAI Python client with DashScope compatible endpoint.
"""

import json
import os
import re
from typing import Dict, Any, Optional

from loguru import logger
from openai import OpenAI
from config.settings import settings

# Global client - initialized on first use
_qwen_client: Optional[OpenAI] = None


def get_qwen_client() -> OpenAI:
    """Get or create OpenAI client for Qwen."""
    global _qwen_client
    
    if _qwen_client is None:
        if not settings.dashscope_api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY environment variable not set. "
                "Please set your Alibaba Qwen API key."
            )
        
        _qwen_client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_endpoint or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        
        logger.info(f"✓ Qwen OpenAI client initialized")
        logger.info(f"✓ Model: {settings.qwen_model}")
        logger.info(f"✓ Endpoint: {settings.dashscope_endpoint or 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'}")
        logger.debug(f"  API Key starts with: {settings.dashscope_api_key[:20]}...")
    
    return _qwen_client


def initialize_qwen():
    """Initialize Qwen API with credentials from settings."""
    if not settings.dashscope_api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY environment variable not set. "
            "Please set your Alibaba Qwen API key."
        )
    
    client = get_qwen_client()
    logger.info(f"✓ Qwen initialized with model: {settings.qwen_model}")
    logger.info(f"✓ API Key length: {len(settings.dashscope_api_key)} chars")
    logger.debug(f"  API Key starts with: {settings.dashscope_api_key[:20]}...")


def call_qwen_with_structured_output(
    system_prompt: str,
    user_message: str,
    response_schema: Dict[str, Any],
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Call Qwen with a structured output request.
    
    Args:
        system_prompt: System message defining the assistant's role
        user_message: User query/prompt
        response_schema: Expected JSON schema for response
        temperature: Sampling temperature (0.0-2.0), defaults to settings
        top_p: Nucleus sampling parameter, defaults to settings
        
    Returns:
        Parsed JSON response matching the schema
        
    Raises:
        ValueError: If API key not set
        RuntimeError: If API call fails or response cannot be parsed
    """
    temperature = temperature or settings.qwen_temperature
    top_p = top_p or settings.qwen_top_p
    
    # Prepare the prompt with schema instructions
    schema_instruction = f"""
You must respond with ONLY valid JSON matching this schema:
{json.dumps(response_schema, indent=2)}

Do not include any text outside the JSON response.
"""
    
    full_user_message = f"{user_message}\n\n{schema_instruction}"
    
    try:
        client = get_qwen_client()
        
        logger.info(f"Calling Qwen ({settings.qwen_model}) with structured output request")
        logger.debug(f"User message: {user_message[:200]}...")
        logger.debug(f"Temperature: {temperature}, Top-p: {top_p}")
        
        response = client.chat.completions.create(
            model=settings.qwen_model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": full_user_message,
                }
            ],
            temperature=temperature,
            top_p=top_p,
        )
        
        logger.debug(f"Qwen response received: {response.choices[0].message.content[:300]}...")
        
        # Extract the response text
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        try:
            # Try to extract JSON if there's surrounding text
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                logger.debug(f"Extracted JSON: {json_str[:200]}...")
            else:
                json_str = response_text
                logger.debug(f"Using full response as JSON: {json_str[:200]}...")
            
            # Validate JSON is complete before parsing
            if not json_str.strip().endswith("}"):
                logger.warning(f"JSON doesn't end with }}, attempting to fix: {json_str[-50:]}")
                # Try to find the last complete object
                brace_count = 0
                last_complete_pos = -1
                for i, char in enumerate(json_str):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            last_complete_pos = i + 1
                
                if last_complete_pos > 0:
                    json_str = json_str[:last_complete_pos]
                    logger.debug(f"Truncated to last complete object at position {last_complete_pos}")
            
            parsed_response = json.loads(json_str)
            logger.debug(f"Parsed response: {parsed_response}")
            return parsed_response
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Qwen JSON response: {response_text[:500]}")
            logger.error(f"JSON parsing error: {str(e)}")
            raise RuntimeError(
                f"Failed to parse Qwen response as JSON: {str(e)}"
            )
        
    except Exception as e:
        logger.error(f"Qwen API call failed: {str(e)}")
        raise RuntimeError(f"Qwen API error: {str(e)}")


def call_qwen_for_triage(user_input: str) -> Dict[str, Any]:
    """
    Use Qwen to perform triage/routing of user query.
    
    Args:
        user_input: User query to route
        
    Returns:
        Dict with keys: ticker, routing_path, confidence, reasoning
    """
    from app.prompts.system_prompts import TRIAGE_PROMPT, SYSTEM_MESSAGE_RESEARCH
    
    schema = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "Stock ticker symbol or empty string"
            },
            "routing_path": {
                "type": "string",
                "enum": ["research", "direct_trade", "general_q"],
                "description": "Routing path for the query"
            },
            "trade_action": {
                "anyOf": [
                    {"type": "string", "enum": ["BUY", "SELL"]},
                    {"type": "null"}
                ],
                "description": "Trade action (BUY/SELL) for direct_trade path, null for others"
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score of routing"
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation of routing decision"
            }
        },
        "required": ["ticker", "routing_path", "confidence", "reasoning"]
    }
    
    formatted_prompt = TRIAGE_PROMPT.format(user_input=user_input)
    
    response = call_qwen_with_structured_output(
        system_prompt=SYSTEM_MESSAGE_RESEARCH,
        user_message=formatted_prompt,
        response_schema=schema,
    )
    
    return response


def call_qwen_for_research(
    ticker: str,
    price: float,
    market_cap: float,
    pe_ratio: float,
    change_pct: float,
) -> str:
    """
    Use Qwen to generate research report.
    
    Args:
        ticker: Stock ticker
        price: Current stock price
        market_cap: Market cap in billions
        pe_ratio: P/E ratio
        change_pct: 52-week change percentage
        
    Returns:
        Research report in markdown format
    """
    from app.prompts.system_prompts import (
        RESEARCH_PROMPT,
        SYSTEM_MESSAGE_RESEARCH
    )
    
    formatted_prompt = RESEARCH_PROMPT.format(
        ticker=ticker,
        price=price,
        market_cap=market_cap,
        pe_ratio=pe_ratio,
        change_pct=change_pct,
    )
    
    try:
        client = get_qwen_client()
        logger.info(f"Calling Qwen for research report on {ticker}")
        
        response = client.chat.completions.create(
            model=settings.qwen_model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE_RESEARCH,
                },
                {
                    "role": "user",
                    "content": formatted_prompt,
                }
            ],
            temperature=settings.qwen_temperature,
            top_p=settings.qwen_top_p,
        )
        
        report = response.choices[0].message.content.strip()
        
        # Sanitize response: fix encoding issues and duplicates
        # Step 1: Replace Unicode asterisks with ASCII
        report = report.replace("∗∗", "**")  # Unicode double asterisk to markdown bold
        report = report.replace("∗", "*")     # Unicode single asterisk to ASCII
        
        # Step 2: Remove corrupted duplicate patterns
        # Fix corrupted bold patterns: "word∗∗word∗∗" → "**word**"
        report = re.sub(r'(\w+)\*\*\1\*\*', r'**\1**', report, flags=re.IGNORECASE)
        
        # Step 3: Clean up lines with obvious duplication
        lines = report.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip empty lines at start
            if not line.strip():
                if cleaned_lines:  # Keep one empty line between content
                    cleaned_lines.append(line)
                continue
            
            # Check if line has obvious duplication pattern
            # If a line contains text repeated exactly, keep only first occurrence
            parts = line.split('**')
            if len(parts) > 2:
                # Rebuild line with proper markdown
                # This handles cases like "text**text**text" → "text**text**"
                reconstructed = parts[0].strip()
                for i in range(1, len(parts), 2):
                    if i < len(parts):
                        bold_text = parts[i].strip()
                        if bold_text and (i + 1 >= len(parts) or parts[i + 1].strip()):
                            reconstructed += f"**{bold_text}**"
                            if i + 1 < len(parts):
                                next_part = parts[i + 1].strip()
                                if next_part and not next_part.startswith(("**", "*")):
                                    reconstructed += f" {next_part}"
                line = reconstructed
            
            cleaned_lines.append(line)
        
        report = "\n".join(cleaned_lines)
        
        # Step 4: Remove excessive blank lines (more than 2 consecutive)
        report = re.sub(r'\n\n\n+', '\n\n', report)
        
        logger.info(f"Research report generated for {ticker}")
        logger.debug(f"Report length (sanitized): {len(report)} characters")
        return report
        
    except Exception as e:
        logger.error(f"Qwen research call failed: {str(e)}")
        raise RuntimeError(f"Qwen research error: {str(e)}")


def call_qwen_for_general_qa(user_input: str) -> str:
    """
    Use Qwen to answer general financial questions.
    
    Args:
        user_input: User question
        
    Returns:
        Answer in markdown format
    """
    from app.prompts.system_prompts import (
        GENERAL_QA_PROMPT,
        SYSTEM_MESSAGE_QA
    )
    
    formatted_prompt = GENERAL_QA_PROMPT.format(user_input=user_input)
    
    try:
        client = get_qwen_client()
        logger.info("Calling Qwen for general Q&A")
        
        response = client.chat.completions.create(
            model=settings.qwen_model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE_QA,
                },
                {
                    "role": "user",
                    "content": formatted_prompt,
                }
            ],
            temperature=settings.qwen_temperature,
            top_p=settings.qwen_top_p,
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Sanitize response: fix encoding issues and duplicates (same as research)
        # Step 1: Replace Unicode asterisks with ASCII
        answer = answer.replace("∗∗", "**")  # Unicode double asterisk to markdown bold
        answer = answer.replace("∗", "*")     # Unicode single asterisk to ASCII
        
        # Step 2: Fix corrupted duplicate patterns
        answer = re.sub(r'(\w+)\*\*\1\*\*', r'**\1**', answer, flags=re.IGNORECASE)
        
        # Step 3: Remove excessive blank lines
        answer = re.sub(r'\n\n\n+', '\n\n', answer)
        
        logger.info("General Q&A response generated")
        logger.debug(f"Answer length (sanitized): {len(answer)} characters")
        return answer
        
    except Exception as e:
        logger.error(f"Qwen QA call failed: {str(e)}")
        raise RuntimeError(f"Qwen QA error: {str(e)}")
