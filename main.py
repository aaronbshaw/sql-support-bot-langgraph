"""
SQL Support Bot - LangGraph Platform Deployment

A customer support chatbot that interacts with SQL database to answer questions
about music and customer accounts using the Chinook database.
"""

import sqlite3
import requests
import json
from functools import partial
from typing import Dict, Any, List, TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, RemoveMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from pydantic import BaseModel, Field
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

# Load environment variables
load_dotenv()

# Initialize LLM
import os
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is required")

model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o", api_key=api_key)

# Database setup
def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)

# Define the memory (short-term memory with thread-level persistence)
memory = MemorySaver()

# Define the state schema for StateGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Tools
@tool
def get_customer_info(identifier: str):
    """Look up customer info by ID (number) or email address. 
    
    Args:
        identifier: Either a customer ID (positive integer) or email address
        
    Examples:
        - get_customer_info("1") - Look up by customer ID
        - get_customer_info("john.doe@email.com") - Look up by email
    """
    try:
        # Check if identifier is a number (customer ID)
        if identifier.isdigit():
            customer_id = int(identifier)
            if customer_id <= 0:
                return f"Error: Customer ID must be a positive integer. You provided: {customer_id}"
            
            # Query by customer ID
            result = db.run(f"SELECT * FROM Customer WHERE CustomerID = {customer_id};")
            
        else:
            # Query by exact email match (escape single quotes to prevent SQL injection)
            email_escaped = identifier.replace("'", "''")
            result = db.run(f"SELECT * FROM Customer WHERE Email = '{email_escaped}';")
        
        # Check if customer was found
        if not result or result.strip() == "":
            return f"No customer found with identifier '{identifier}'. Please check the ID or email and try again."
        
        return result
        
    except Exception as e:
        return f"Error retrieving customer information: {str(e)}. Please try again or contact support."

@tool
def update_customer_info(customer_id: str, field_name: str, new_value: str):
    """Update a customer's information in the database.
    
    Args:
        customer_id: The customer ID (must be a positive integer)
        field_name: The field to update. Valid fields are: FirstName, LastName, Company, 
                   Address, City, State, Country, PostalCode, Phone, Fax, Email
        new_value: The new value for the field
        
    Examples:
        - update_customer_info("1", "Phone", "+1 555-0123")
        - update_customer_info("5", "Email", "newemail@example.com")
        - update_customer_info("10", "Address", "123 New Street")
    
    Returns:
        Success message with updated info, or error message if update failed
    """
    try:
        # Validate customer_id is a number
        if not customer_id.isdigit():
            return f"Error: Customer ID must be a number. You provided: {customer_id}"
        
        cid = int(customer_id)
        if cid <= 0:
            return f"Error: Customer ID must be a positive integer. You provided: {cid}"
        
        # Validate field_name (only allow specific fields, NOT CustomerId)
        valid_fields = [
            "FirstName", "LastName", "Company", "Address", "City", 
            "State", "Country", "PostalCode", "Phone", "Fax", "Email"
        ]
        
        if field_name not in valid_fields:
            return f"Error: Invalid field name '{field_name}'. Valid fields are: {', '.join(valid_fields)}"
        
        # First, verify customer exists
        check_result = db.run(f"SELECT CustomerId FROM Customer WHERE CustomerId = {cid};")
        if not check_result or check_result.strip() == "":
            return f"Error: No customer found with ID {cid}. Cannot update."
        
        # Escape single quotes in new_value to prevent SQL injection
        escaped_value = new_value.replace("'", "''")
        
        # Perform the update
        update_query = f"UPDATE Customer SET {field_name} = '{escaped_value}' WHERE CustomerId = {cid};"
        db.run(update_query)
        
        # Verify the update by fetching the updated record
        result = db.run(f"SELECT * FROM Customer WHERE CustomerId = {cid};")
        
        return f"Successfully updated {field_name} for customer ID {cid}. Updated customer info:\n{result}"
        
    except Exception as e:
        return f"Error updating customer info: {str(e)}"

@tool
def get_albums_by_artist(artist: str):
    """Get albums by an artist."""
    # Escape single quotes to prevent SQL injection
    artist_escaped = artist.replace("'", "''")
    return db.run(
        f"""
        SELECT Album.Title, Artist.Name 
        FROM Album 
        JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        WHERE Artist.Name LIKE '%{artist_escaped}%';
        """,
        include_columns=True
    )

@tool
def get_tracks_by_artist(artist: str):
    """Get songs by an artist (or similar artists)."""
    # Escape single quotes to prevent SQL injection
    artist_escaped = artist.replace("'", "''")
    return db.run(
        f"""
        SELECT Track.Name as SongName, Artist.Name as ArtistName 
        FROM Album 
        LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
        LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
        WHERE Artist.Name LIKE '%{artist_escaped}%';
        """,
        include_columns=True
    )

@tool
def check_for_songs(song_title):
    """Check if a song exists by its name."""
    # Escape single quotes to prevent SQL injection
    song_escaped = song_title.replace("'", "''")
    return db.run(
        f"""
        SELECT * FROM Track WHERE Name LIKE '%{song_escaped}%';
        """,
        include_columns=True
    )

@tool
def get_songs_in_playlist(playlist_name: str):
    """Return songs for a playlist by (partial) name match."""
    # Escape single quotes to prevent SQL injection
    name_escaped = playlist_name.replace("'", "''")
    return db.run(
        f"""
        SELECT p.Name as PlaylistName, t.Name as SongName
        FROM Playlist p
        JOIN PlaylistTrack pt ON p.PlaylistId = pt.PlaylistId
        JOIN Track t ON t.TrackId = pt.TrackId
        WHERE p.Name LIKE '%{name_escaped}%'
        ORDER BY p.Name, t.Name;
        """,
        include_columns=True
    )

# Router model
class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""
    choice: str = Field(description="should be one of: music, customer")

# Prompts
customer_prompt = """Your job is to help a user with their account information.

IMPORTANT: Always review the conversation history AND conversation summary (if present) to understand what the customer has asked about previously. You can reference previous topics, questions, or information they mentioned.

**CRITICAL: Check the conversation summary for customer ID or email address BEFORE asking the user for it again.**
- If the summary contains "Customer ID: [number]" or "ID: [number]", use that ID
- If the summary contains an email address, use that email
- Only ask for ID/email if it's NOT in the summary or previous messages

You have access to two customer tools:

1. get_customer_info - Look up customer information by:
   - Customer ID (a positive number like "1", "5", "10")
   - Email address (exact match like "john.doe@email.com")

2. update_customer_info - Update customer information fields:
   - Valid fields: FirstName, LastName, Company, Address, City, State, Country, PostalCode, Phone, Fax, Email
   - Requires: customer_id, field_name, new_value
   - Example: update_customer_info("1", "Phone", "+1 555-0123")

IMPORTANT GUIDELINES:
1. **First, check conversation summary and history for customer ID or email**
2. Only ask for customer ID/email if not already provided in summary or context
3. If the user doesn't know their customer ID, ask for their email address instead
4. Before updating, verify the customer ID and confirm what field they want to update
5. After updating, show the user their updated information (do NOT call update again)
6. If there's an error or no customer found, explain the issue clearly
7. Email addresses must be exact matches - partial matches will not work

If you are unable to help the user, politely explain what information you need and suggest they contact support if needed."""

song_system_message = """Your job is to help a customer find information about music they are looking for. 

IMPORTANT: Always review the conversation history to understand what the customer has asked about previously. You can reference previous artists, songs, or topics they mentioned.

You only have certain tools you can use. If a customer asks you to look something up that you don't know how, politely tell them what you can help with.

When looking up artists and songs, sometimes the artist/song will not be found. In that case, the tools will return information \
on simliar songs and artists. This is intentional, it is not the tool messing up."""

system_message = """Your job is to help as a customer service representative for a music store.

You have TWO options for each customer request:

1. **RESPOND DIRECTLY** if you can answer from the conversation history or if the question or comment is not related to music or their account.
2. **ROUTE TO SPECIALIST** if you need specialist help related to music or their account.

## When to RESPOND DIRECTLY:
- Customer asks about previous topics mentioned in conversation
- Customer asks for reminders or clarifications about what was discussed
- Customer asks follow-up questions you can answer from context
- Customer asks "who was the last band we talked about?" or similar memory questions

## When to ROUTE TO SPECIALIST:
- Customer asks about music, songs, albums, artists, playlists → Router(choice="music")
- Customer asks about their account, profile, customer information → Router(choice="customer")
- Customer provides new information that needs specialist processing

## Smart Context Passing:
When routing to specialists, include relevant context from the conversation:
- If customer previously mentioned a customer ID or email, pass that context
- If customer previously asked about a specific artist, include that context
- Always provide the specialist with the information they need

## Examples:
- "find songs by U2" → Router(choice="music")
- "who was the last band we talked about?" → Respond directly: "We were discussing AC/DC"
- "what's my email?" → Router(choice="customer") with context: "Customer previously provided ID: 3"
- "tell me about my account again" → Router(choice="customer") with context: "Customer ID: 3 from previous conversation"

IMPORTANT: Use Router tool when you need specialist help, respond directly when you can answer from conversation history.

When tools have been called and you receive tool responses, you must:
1) If the tool output answers the user's request, reply to the user in clear natural language using the tool results. Do NOT route again - you have the answer.
2) If the tool output is insufficient to answer the user, ask a concise follow-up question for the exact missing information needed. Only route again if a different specialist is required.

CRITICAL: If you see "Tool results summary" in your context, you already have tool results and should respond directly to the user. Do NOT route to specialists again.

Routing rules:
- ALWAYS base your routing decision on the MOST RECENT user message, even if previous conversation was about a different topic
- Users can switch topics at any time - respect the topic in their latest message
- Emit EXACTLY ONE call to the Router tool (choose one of: music OR customer). Never call both
- Only respond directly (without routing) if the latest message is a simple clarification about what was JUST discussed

When routing to a specialist, include the user's exact request in your assistant message content so the specialist knows what to help with. Format: "User request: [exact user message]"""

# Chains
def get_customer_messages(messages):
    return [SystemMessage(content=customer_prompt)] + messages

def get_song_messages(messages):
    return [SystemMessage(content=song_system_message)] + messages

def get_messages(messages):
    return [SystemMessage(content=system_message)] + messages

customer_chain = get_customer_messages | model.bind_tools([get_customer_info, update_customer_info])
song_recc_chain = get_song_messages | model.bind_tools([get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_songs_in_playlist])
general_chain = get_messages | model.bind_tools([Router])

# Helper functions
def add_name(message, name):
    _dict = message.model_dump()
    _dict["name"] = name
    return AIMessage(**_dict)

def _get_last_ai_message(messages):
    for m in messages[::-1]:
        if isinstance(m, AIMessage):
            return m
    return None

def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs

def should_summarize(state):
    """Determine if we should summarize the conversation.
    
    Returns 'summarize' if we have 6+ meaningful messages (excluding summary itself),
    otherwise returns 'continue' to proceed with normal flow.
    
    This implements PROGRESSIVE summarization - the summary is continuously updated
    as new messages arrive, not just created once.
    """
    messages = state["messages"]
    
    # Get meaningful messages in chronological order (same logic as summarize_conversation)
    meaningful_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            meaningful_messages.append(msg)
        elif isinstance(msg, AIMessage):
            # Exclude AI messages with tool calls (routing artifacts)
            if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                if not (hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs):
                    meaningful_messages.append(msg)
    
    # Summarize whenever we have more than 6 meaningful messages
    # This creates a rolling/progressive summary that updates continuously
    if len(meaningful_messages) > 6:
        return "summarize"
    else:
        return "continue"

def _route(state):
    messages = state["messages"]
    if not messages:
        return "general"
    
    last_message = messages[-1]
    
    # If last message is a human message, route to general
    if isinstance(last_message, HumanMessage):
        return "general"
    
    # If last message is an AI message with tool calls
    if isinstance(last_message, AIMessage) and _is_tool_call(last_message):
        if last_message.name == "general":
            # General agent made a routing decision
            # Check if we just came from a specialist agent WITHOUT new user input
            # If so, don't route to specialists again - END instead to avoid loops
            if len(messages) >= 2:
                prev_msg = messages[-2]
                # Check if previous message is from specialist and there's no HumanMessage after it
                if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, 'name') and prev_msg.name in ["music", "customer"]:
                    # Find if there's a HumanMessage between the specialist response and now
                    # Look back through recent messages to see if there's new user input
                    has_new_user_input = False
                    for i in range(len(messages) - 1, max(0, len(messages) - 5), -1):
                        if isinstance(messages[i], HumanMessage):
                            # Found a human message - check if it's after the specialist response
                            specialist_idx = messages.index(prev_msg) if prev_msg in messages else -1
                            human_idx = i
                            if human_idx > specialist_idx:
                                has_new_user_input = True
                                break
                            break
                    
                    # If no new user input since specialist responded, END to avoid loop
                    if not has_new_user_input:
                        return END
            
            tool_calls = last_message.additional_kwargs['tool_calls']
            if len(tool_calls) > 1:
                # If multiple tool calls, just take the first one
                tool_call = tool_calls[0]
            else:
                tool_call = tool_calls[0]
            choice = json.loads(tool_call['function']['arguments'])['choice']
            return choice
        elif last_message.name == "music":
            # Music agent made tool calls - route to music_tools
            return "music_tools"
        elif last_message.name == "customer":
            # Customer agent made tool calls - check if sensitive or safe
            tool_calls = last_message.additional_kwargs.get('tool_calls', [])
            if tool_calls:
                tool_call = tool_calls[0]
                tool_name = tool_call['function']['name']
                
                # Check if this is a sensitive operation
                if tool_name == "update_customer_info":
                    # Route to sensitive tools (interrupt will pause before execution)
                    return "customer_sensitive_tools"
                else:
                    # Safe operation - go directly to safe tools
                    return "customer_safe_tools"
            return "customer_safe_tools"  # Default to safe tools
        else:
            # Unknown agent with tool calls
            return "general"
    
    # If last message is a tool response, route back to the agent that called it
    if isinstance(last_message, ToolMessage):
        # Look back to find which agent made the tool call
        for i in range(len(messages) - 2, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage) and _is_tool_call(msg):
                agent_name = getattr(msg, 'name', None)
                if agent_name == "music":
                    return "music"
                elif agent_name == "customer":
                    return "customer"
                break
        # Default to general if we can't determine the calling agent
        return "general"
    
    # If last message is an AI message without tool calls, decide routing based on agent
    if isinstance(last_message, AIMessage) and not _is_tool_call(last_message):
        if last_message.name == "general":
            return END  # General can end the turn
        elif last_message.name in ["music", "customer"]:
            # Specialist agents without tool calls have responded with text
            # This means they're either asking for more info or can't help
            # END the turn so user can respond
            return END
    
    # Default fallback
    return "general"

def _filter_out_routes(messages):
    """Filter out routing messages but keep tool calls and responses."""
    ms = []
    for m in messages:
        if _is_tool_call(m) and m.name == "general":
            # Skip general agent routing messages
            continue
        ms.append(m)
    return ms

def _filter_incomplete_tool_calls(messages):
    """Filter out incomplete tool call sequences to avoid OpenAI errors."""
    if not messages:
        return messages
    
    # Only pass the last human message to avoid recursion loops
    # The checkpointer maintains full conversation history, but we only need current context
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if human_messages:
        return [human_messages[-1]]  # Only the most recent human message
    
    return messages

# Node definitions
music_tools = [get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_songs_in_playlist]
customer_safe_tools = [get_customer_info]
customer_sensitive_tools = [update_customer_info]   # This tool is only used for sensitive information like updating email or phone number

def summarize_conversation(state):
    """Progressively summarize conversation to keep context manageable.
    
    This node removes old messages and replaces them with a rolling summary.
    If a summary already exists, it EXTENDS it with new information.
    Based on LangGraph tutorial: https://colab.research.google.com/drive/106khdPGF85ea5NlU9Xer-g7gTZ0eLD66
    """
    messages = state["messages"]
    
    # Check if we have an existing summary
    existing_summary = None
    for msg in messages:
        if isinstance(msg, SystemMessage) and hasattr(msg, 'name') and msg.name == "conversation_summary":
            existing_summary = msg
            break
    
    # Get meaningful messages (human + AI responses) in CHRONOLOGICAL ORDER
    # IMPORTANT: Preserve original order so last 6 includes a balanced mix
    meaningful_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            meaningful_messages.append(msg)
        elif isinstance(msg, AIMessage):
            # Exclude AI messages with tool calls (routing artifacts)
            if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                if not (hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs):
                    meaningful_messages.append(msg)
    
    # Keep the last 6 meaningful messages, summarize/extend with the rest
    # By preserving chronological order, we get a balanced mix (e.g., H, A, H, A, H, A)
    messages_to_keep = meaningful_messages[-6:]
    messages_to_summarize = [m for m in meaningful_messages if m not in messages_to_keep]
    
    if not messages_to_summarize:
        # Nothing to summarize
        return {"messages": []}
    
    # Build summary text for new messages
    # Include ToolMessages in the summary text but DON'T delete them from state
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    
    summary_parts = []
    for msg in messages_to_summarize:
        if isinstance(msg, HumanMessage):
            summary_parts.append(f"User: {msg.content[:200]}")
        elif isinstance(msg, AIMessage):
            content = msg.content[:200] if msg.content else "[no content]"
            summary_parts.append(f"Bot: {content}")
    
    # Add relevant tool results to summary (especially customer lookups)
    for tm in tool_messages:
        tool_name = getattr(tm, 'name', '')
        if tool_name == 'get_customer_info':
            # Extract customer ID from result
            content_str = str(tm.content)
            summary_parts.append(f"[Customer Info Retrieved]: {content_str[:150]}")
    
    new_messages_text = "\n".join(summary_parts)
    
    # Create or extend summary
    if existing_summary:
        # EXTEND existing summary with new messages
        summary_prompt = f"""You are updating a conversation summary for a music store support bot.

EXISTING SUMMARY:
{existing_summary.content}

NEW MESSAGES TO ADD TO SUMMARY:
{new_messages_text}

Create an UPDATED comprehensive summary that:
- Combines information from the existing summary with the new messages
- **CRITICAL: If a customer ID or email address was provided, ALWAYS include it in the summary**
- Format customer info clearly: "Customer: [Name], ID: [number], Email: [email]" if available
- Maintains ALL important details (customer IDs, names, songs, albums, playlists, artists discussed)
- Keeps the summary current and factual
- Does NOT lose any critical information from either the old summary or new messages

Updated Summary:"""
        
        # Get updated summary
        summary_response = model.invoke([SystemMessage(content=summary_prompt)])
        
        # Delete the old summary and old human/AI messages
        # IMPORTANT: Do NOT delete ToolMessages - they contain critical data (customer lookups, etc.)
        delete_messages = [RemoveMessage(id=existing_summary.id)] + [RemoveMessage(id=m.id) for m in messages_to_summarize]
    else:
        # CREATE first summary
        summary_prompt = f"""You are summarizing a conversation between a user and a music store support bot.

Create a comprehensive summary (4-6 sentences) covering:
- ALL topics discussed (music queries, customer account questions, etc.)
- **CRITICAL: If a customer ID or email address was provided, ALWAYS include it at the START of the summary**
- Format customer info clearly: "Customer: [Name], ID: [number], Email: [email]" if available
- ALL important information provided (customer IDs, email addresses, artist names, album requests)
- Specific data retrieved (customer names, song lists, etc.)
- Any pending actions or unresolved requests
- Current status or outcomes

Include ALL factual details that might be needed for future turns. Do NOT omit customer IDs, information, songs, albums, playlists, artists, or other key data.

Conversation:
{new_messages_text}

Comprehensive Summary:"""
        
        # Get summary from model
        summary_response = model.invoke([SystemMessage(content=summary_prompt)])
        
        # Delete old human/AI messages only (NOT ToolMessages - they're preserved)
        delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize]
    
    # Create the summary message with clear formatting
    summary_message = SystemMessage(
        content=f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVERSATION HISTORY SUMMARY (For Background Context)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{summary_response.content}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOTE: The summary above is HISTORICAL CONTEXT ONLY.
Recent conversation messages follow below in chronological order.
The LAST user message in the conversation is the CURRENT REQUEST to respond to.
Earlier messages provide context but should NOT influence routing decisions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""",
        name="conversation_summary"
    )
    
    # Return: new summary + delete old messages (and old summary if it existed)
    return {"messages": [summary_message] + delete_messages}

# Tool nodes
music_tools_node = ToolNode(music_tools)
customer_safe_tools_node = ToolNode(customer_safe_tools)
customer_sensitive_tools_node = ToolNode(customer_sensitive_tools)

def general_node(state):
    messages = state["messages"]
    
    # For general agent, pass full conversation context for smart routing and responses
    # Include all human messages and recent safe AI responses (no tool calls)
    # Do NOT include raw ToolMessage objects (they must follow tool_calls). Instead, summarize them.
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    
    # Filter out AI messages with tool calls to avoid OpenAI errors
    safe_ai_messages = []
    for ai_msg in ai_messages:
        if not (hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls):
            safe_ai_messages.append(ai_msg)
    
    # Build context (don't add system_message here - the chain will add it)
    conversation_context = []
    
    # Check if we have an existing conversation summary (created by summarize_conversation node)
    existing_summary = None
    for msg in messages:
        if isinstance(msg, SystemMessage) and hasattr(msg, 'name') and msg.name == "conversation_summary":
            existing_summary = msg
            break
    
    # If there's a summary, include it for background context
    if existing_summary:
        conversation_context.append(existing_summary)
    
    # Include messages in CHRONOLOGICAL ORDER (as they appear in state)
    # This preserves conversation flow: H1→A1→H2→A2→H3→A3
    for msg in messages:
        # Skip summary messages (already added above)
        if isinstance(msg, SystemMessage) and hasattr(msg, 'name') and msg.name == "conversation_summary":
            continue
        # Add human messages
        elif isinstance(msg, HumanMessage):
            conversation_context.append(msg)
        # Add AI messages (but not routing/tool calls)
        elif isinstance(msg, AIMessage):
            if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                if not (hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs):
                    conversation_context.append(msg)
    
    # Add clear instruction about which message to respond to
    if human_messages:
        conversation_context.append(
            SystemMessage(content=f"⚠️ ROUTING INSTRUCTION: The LAST user message above is the CURRENT request. Route based on THAT message. Earlier messages are for context only.")
        )
    
    # Summarize recent tool outputs for safe inclusion
    if tool_messages:
        recent_tools = tool_messages[-5:] if len(tool_messages) > 5 else tool_messages
        # Create a compact summary string of tool outputs
        tool_summaries = []
        for tm in recent_tools:
            name = getattr(tm, "name", "tool")
            content = tm.content if hasattr(tm, "content") and isinstance(tm.content, str) else str(tm.content)
            # Truncate overly long tool content
            if len(content) > 800:
                content = content[:800] + "..."
            tool_summaries.append(f"{name}: {content}")
        tools_summary_text = "\n".join(tool_summaries)
        conversation_context.append(SystemMessage(content=f"Tool results summary (most recent first):\n{tools_summary_text}"))
    
    result = general_chain.invoke(conversation_context)
    return {"messages": [add_name(result, name="general")]}

def music_node(state):
    messages = state["messages"]
    
    # Extract message types
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    
    # Build context (don't add system message here - the chain will add it)
    conversation_context = []
    
    # Check if there's a conversation summary (for context like "these artists")
    existing_summary = None
    for msg in messages:
        if isinstance(msg, SystemMessage) and hasattr(msg, 'name') and msg.name == "conversation_summary":
            existing_summary = msg
            break
    
    # If there's a summary, include it so agent can reference previous context
    if existing_summary:
        conversation_context.append(existing_summary)
    
    # Include recent human messages (last 3) for references like "these artists"
    # This allows the agent to understand pronouns and references
    recent_humans = human_messages[-3:] if len(human_messages) > 3 else human_messages
    for hm in recent_humans:
        conversation_context.append(hm)
    
    # Only include tool results from the CURRENT turn (after the last human message)
    # Find the index of the most recent human message
    if human_messages:
        last_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        
        # Get tool messages that came after the last human message
        if last_human_idx is not None:
            current_turn_tools = [m for m in messages[last_human_idx:] if isinstance(m, ToolMessage)]
            
            # Include these tool results for formulating response
            for tm in current_turn_tools:
                content = tm.content if hasattr(tm, 'content') and isinstance(tm.content, str) else str(tm.content)
                if len(content) > 1000:
                    content = content[:1000] + "..."
                tool_name = getattr(tm, "name", "tool")
                conversation_context.append(SystemMessage(content=f"Tool result from {tool_name}: {content}"))
    
    # Add clear instruction about current vs context messages
    if human_messages and len(human_messages) > 1:
        conversation_context.append(
            SystemMessage(content=f"⚠️ INSTRUCTION: The LAST user message above is the CURRENT request to answer. Earlier messages provide context (for references like 'these artists').")
        )
    
    result = song_recc_chain.invoke(conversation_context)
    return {"messages": [add_name(result, name="music")]}

def customer_node(state):
    messages = state["messages"]
    
    # For customer agent, pass recent conversation context for better responses
    # Include recent human messages and safe AI responses (no tool calls)
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    
    # Filter out AI messages with tool calls to avoid OpenAI errors
    safe_ai_messages = []
    for ai_msg in ai_messages:
        if not (hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls):
            safe_ai_messages.append(ai_msg)
    
    # Build context (don't add system message here - the chain will add it)
    conversation_context = []
    
    # Check if there's a conversation summary and include it first
    existing_summary = None
    for msg in messages:
        if isinstance(msg, SystemMessage) and hasattr(msg, 'name') and msg.name == "conversation_summary":
            existing_summary = msg
            break
    
    if existing_summary:
        conversation_context.append(existing_summary)
    
    # Include messages in CHRONOLOGICAL ORDER to preserve conversation flow
    for msg in messages:
        # Skip summary (already added)
        if isinstance(msg, SystemMessage) and hasattr(msg, 'name') and msg.name == "conversation_summary":
            continue
        # Add human messages
        elif isinstance(msg, HumanMessage):
            conversation_context.append(msg)
        # Add AI messages (exclude tool calls)
        elif isinstance(msg, AIMessage):
            if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                if not (hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs):
                    conversation_context.append(msg)
    
    # Always include the most recent customer lookup for context
    # This ensures the agent remembers customer info even across multiple turns
    customer_info_tools = [
        m for m in messages 
        if isinstance(m, ToolMessage) and getattr(m, 'name', '') == 'get_customer_info'
    ]
    if customer_info_tools:
        # Include the most recent customer info lookup
        most_recent_customer_lookup = customer_info_tools[-1]
        content = most_recent_customer_lookup.content
        if isinstance(content, str) and len(content) > 500:
            content = content[:500] + "..."
        conversation_context.append(
            SystemMessage(content=f"Previously fetched customer info: {content}")
        )
    
    # Include tool results from the CURRENT turn (after the last human message)
    if human_messages:
        last_human_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                last_human_idx = i
                break
        
        # Get tool messages that came after the last human message
        if last_human_idx is not None:
            current_turn_tools = [m for m in messages[last_human_idx:] if isinstance(m, ToolMessage)]
            
            # Include these tool results for formulating response
            for tm in current_turn_tools:
                # Skip if we already included this as the most recent customer lookup
                if tm == customer_info_tools[-1] if customer_info_tools else None:
                    continue
                    
                content = tm.content if hasattr(tm, 'content') and isinstance(tm.content, str) else str(tm.content)
                if len(content) > 1000:
                    content = content[:1000] + "..."
                tool_name = getattr(tm, "name", "tool")
                conversation_context.append(SystemMessage(content=f"Tool result from {tool_name}: {content}"))
    
    # Check if we just executed update_customer_info to prevent loops
    recent_update_tools = [
        m for m in messages[-3:] 
        if isinstance(m, ToolMessage) and getattr(m, 'name', '') == 'update_customer_info'
    ]
    
    if recent_update_tools:
        # We just executed an update - tell the LLM to format a response, not call update again
        conversation_context.append(
            SystemMessage(content=f"⚠️ IMPORTANT: An update was just completed. Format a friendly response for the user based on the tool result above. Do NOT call update_customer_info again - the update is already done.")
        )
    
    # Add clear instruction about current vs context messages
    if human_messages:
        conversation_context.append(
            SystemMessage(content=f"⚠️ INSTRUCTION: The LAST user message above is the CURRENT request to address. Check the conversation summary and earlier messages for customer ID/email before asking for it again.")
        )
    
    result = customer_chain.invoke(conversation_context)
    return {"messages": [add_name(result, name="customer")]}

# Graph definition
def create_graph():
    """Create and return the LangGraph workflow."""
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("summarize_conversation", summarize_conversation)
    workflow.add_node("general", general_node)
    workflow.add_node("music", music_node)
    workflow.add_node("customer", customer_node)
    workflow.add_node("music_tools", music_tools_node)
    workflow.add_node("customer_safe_tools", customer_safe_tools_node)
    workflow.add_node("customer_sensitive_tools", customer_sensitive_tools_node)
    
    # Add edges with proper routing restrictions
    # Start with conditional: check if we need to summarize first
    workflow.add_conditional_edges(
        START,
        should_summarize,
        {
            "summarize": "summarize_conversation",
            "continue": "general"
        }
    )
    
    # After summarization, go to general
    workflow.add_edge("summarize_conversation", "general")
    
    # General routes to specialists or ends
    workflow.add_conditional_edges("general", _route, {"music": "music", "customer": "customer", END: END})
    
    # Specialist agents can route to their tools, back to themselves, to general, or END
    workflow.add_conditional_edges("music", _route, {"music_tools": "music_tools", "general": "general", END: END})
    workflow.add_conditional_edges("customer", _route, {
        "customer_safe_tools": "customer_safe_tools", 
        "customer_sensitive_tools": "customer_sensitive_tools",
        "general": "general", 
        END: END
    })
    
    # Tools route back to their specialist agent
    workflow.add_conditional_edges("music_tools", _route, {"music": "music"})
    workflow.add_conditional_edges("customer_safe_tools", _route, {"customer": "customer"})
    workflow.add_conditional_edges("customer_sensitive_tools", _route, {"customer": "customer"})
    
    # Compile with checkpointer for thread-level persistence
    return workflow.compile(checkpointer=memory)

# Create the graph instance
graph = create_graph()

# For LangGraph Platform deployment
def get_graph():
    """Return the compiled graph for LangGraph Platform."""
    return graph

if __name__ == "__main__":
    # Debug mode - set to True to run tests, False to interact with the bot
    DEBUG_MODE = False
    
    if DEBUG_MODE:
        # Test customer follow-up scenario
        from langchain_core.messages import HumanMessage
        
        # Recreate graph to ensure latest workflow changes are used
        graph = create_graph()
        
        print("🧪 Testing Customer Follow-up Context")
        print("=" * 50)
        
        # Test: Account question followed by providing ID
        print("\n🧪 Testing: Account Question → Providing ID")
        config = {"configurable": {"thread_id": "test-customer-followup"}, "recursion_limit": 25}
        
        try:
            # Step 1: Ask about account information
            print("\n📨 Step 1: User asks about account information")
            test_input_1 = {"messages": [HumanMessage(content="What's my account information?")]}
            result_1 = graph.invoke(test_input_1, config=config)
            print(f"   Response messages: {len(result_1['messages'])}")
            
            # Show step 1 messages
            for i, msg in enumerate(result_1['messages']):
                msg_type = type(msg).__name__
                msg_name = getattr(msg, 'name', 'no_name')
                msg_content = ""
                if hasattr(msg, 'content') and msg.content:
                    msg_content = msg.content[:80] if isinstance(msg.content, str) else str(msg.content)[:80]
                
                # Check for tool calls
                if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                    tool_calls = msg.additional_kwargs['tool_calls']
                    print(f"   {i+1}. {msg_name}: {msg_type} (has {len(tool_calls)} tool calls)")
                    for tc in tool_calls:
                        print(f"       └─ Tool: {tc['function']['name']} | Args: {tc['function']['arguments'][:50]}...")
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"   {i+1}. {msg_name}: {msg_type} (has {len(msg.tool_calls)} tool calls)")
                else:
                    print(f"   {i+1}. {msg_name}: {msg_type}")
                    if msg_content:
                        print(f"       └─ {msg_content}...")
            
            # Step 2: Provide customer ID
            print("\n📨 Step 2: User provides customer ID")
            test_input_2 = {"messages": [HumanMessage(content="My customer ID is 1")]}
            result_2 = graph.invoke(test_input_2, config=config)
            print(f"   Response messages: {len(result_2['messages'])}")
            
            # Show step 2 messages
            for i, msg in enumerate(result_2['messages']):
                msg_type = type(msg).__name__
                msg_name = getattr(msg, 'name', 'no_name')
                msg_content = ""
                if hasattr(msg, 'content') and msg.content:
                    msg_content = msg.content[:80] if isinstance(msg.content, str) else str(msg.content)[:80]
                
                # Check for tool calls
                if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                    tool_calls = msg.additional_kwargs['tool_calls']
                    print(f"   {i+1}. {msg_name}: {msg_type} (has {len(tool_calls)} tool calls)")
                    for tc in tool_calls:
                        print(f"       └─ Tool: {tc['function']['name']} | Args: {tc['function']['arguments'][:50]}...")
                elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"   {i+1}. {msg_name}: {msg_type} (has {len(msg.tool_calls)} tool calls)")
                else:
                    print(f"   {i+1}. {msg_name}: {msg_type}")
                    if msg_content:
                        print(f"       └─ {msg_content}...")
            
            # Check final response
            last_msg = result_2['messages'][-1]
            if hasattr(last_msg, 'content') and last_msg.content:
                print(f"\n   📝 Final response: {last_msg.content[:150]}...")
            
            print("\n✅ Customer follow-up: PASSED")
                
        except Exception as e:
            print(f"\n❌ Customer follow-up: FAILED - {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("Customer follow-up test complete!")
        
        # Test music query to check for duplicate responses
        print("\n" + "=" * 50)
        print("🧪 Testing Music Query - No Duplicates")
        print("=" * 50)
        
        # Create a fresh thread for this test
        music_config = {"configurable": {"thread_id": "test-music-no-duplicates"}, "recursion_limit": 25}
        
        try:
            # Step 1: Ask about U2
            print("\n📨 Step 1: User asks about U2 songs")
            test_music_1 = {"messages": [HumanMessage(content="Show me some songs by U2")]}
            result_m1 = graph.invoke(test_music_1, config=music_config)
            last_msg_1 = result_m1['messages'][-1]
            if hasattr(last_msg_1, 'content'):
                response_1 = last_msg_1.content
                print(f"   Response length: {len(response_1)} chars")
                print(f"   Preview: {response_1[:200]}...")
            
            # Step 2: Ask about AC/DC
            print("\n📨 Step 2: User asks about AC/DC songs")
            test_music_2 = {"messages": [HumanMessage(content="Now show me songs by AC/DC")]}
            result_m2 = graph.invoke(test_music_2, config=music_config)
            last_msg_2 = result_m2['messages'][-1]
            if hasattr(last_msg_2, 'content'):
                response_2 = last_msg_2.content
                print(f"   Response length: {len(response_2)} chars")
                print(f"   Preview: {response_2[:200]}...")
                
                # Check if U2 is mentioned in the AC/DC response (it shouldn't be)
                if "U2" in response_2 or "u2" in response_2.lower():
                    print("\n   ⚠️  WARNING: U2 mentioned in AC/DC response - possible duplicate issue")
                else:
                    print("\n   ✅ No duplicate data - U2 not mentioned in AC/DC response")
            
            print("\n✅ Music query test: PASSED")
                
        except Exception as e:
            print(f"\n❌ Music query test: FAILED - {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("Music query test complete!")
        
        # Test customer update functionality
        print("\n" + "=" * 50)
        print("🧪 Testing Customer Update")
        print("=" * 50)
        
        # Create a fresh thread for this test
        update_config = {"configurable": {"thread_id": "test-customer-update"}, "recursion_limit": 50}
        
        try:
            # Step 1: Get current customer info
            print("\n📨 Step 1: Get current customer info (ID 1)")
            test_get = {"messages": [HumanMessage(content="Show me the info for customer ID 1")]}
            result_get = graph.invoke(test_get, config=update_config)
            last_msg_get = result_get['messages'][-1]
            if hasattr(last_msg_get, 'content'):
                print(f"   Preview: {last_msg_get.content[:200]}...")
            
            # Step 2: Update the phone number
            print("\n📨 Step 2: Update phone number for customer 1")
            test_update = {"messages": [HumanMessage(content="Please update the phone number for customer 1 to +1 555-TEST-123")]}
            result_update = graph.invoke(test_update, config=update_config)
            last_msg_update = result_update['messages'][-1]
            if hasattr(last_msg_update, 'content'):
                response_update = last_msg_update.content
                print(f"   Response length: {len(response_update)} chars")
                print(f"   Preview: {response_update[:300]}...")
                
                # Check if update was successful
                if "successfully updated" in response_update.lower() or "555-TEST-123" in response_update:
                    print("\n   ✅ Update successful - new phone number appears in response")
                else:
                    print("\n   ⚠️  WARNING: Update may have failed - check response")
            
            print("\n✅ Customer update test: PASSED")
                
        except Exception as e:
            print(f"\n❌ Customer update test: FAILED - {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 50)
        print("Customer update test complete!")
    
    else:
        # Interactive mode - chat with the bot
        from langchain_core.messages import HumanMessage
        
        print("🤖 SQL Support Bot - Interactive Mode")
        print("=" * 50)
        print("Ask me about music or your account information!")
        print("Type 'quit' or 'exit' to stop.")
        print("=" * 50)
        
        # Create a thread for tracking the state of our run
        import uuid
        thread_id = f"session-{uuid.uuid4().hex[:8]}"
        print(f"📝 Starting conversation thread: {thread_id}")
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n👋 Goodbye! Thanks for using SQL Support Bot!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Process the message - checkpointer automatically maintains conversation history
                graph_input = {"messages": [HumanMessage(content=user_input)]}
                config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 20}
                
                result = graph.invoke(graph_input, config=config)
                
                # Show the bot's response
                if result and result.get('messages'):
                    last_msg = result['messages'][-1]
                    bot_name = getattr(last_msg, 'name', None)
                    if bot_name == "general":
                        bot_label = "General Bot"
                    elif bot_name == "music":
                        bot_label = "Music Bot"
                    elif bot_name == "customer":
                        bot_label = "Customer Bot"
                    else:
                        bot_label = "Bot"
                    if hasattr(last_msg, 'content') and last_msg.content:
                        print(f"\n🤖 {bot_label}: {last_msg.content}")
                    else:
                        print(f"\n🤖 {bot_label}: [Processing your request...]")
                else:
                    print("\n🤖 Bot: I'm not sure how to help with that. Try asking about music or your account!")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for using SQL Support Bot!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again or contact support.")
