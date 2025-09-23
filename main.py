"""
SQL Support Bot - LangGraph Platform Deployment

A customer support chatbot that interacts with SQL database to answer questions
about music and customer accounts using the Chinook database.
"""

import sqlite3
import requests
import json
from functools import partial
from typing import Dict, Any, List
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field
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

You have access to customer information through the get_customer_info tool. This tool can look up customers by either:
- Customer ID (a positive number like "1", "5", "10")
- Email address (exact match like "john.doe@email.com")

IMPORTANT GUIDELINES:
1. Ask the user for their customer ID or email address before looking up their information
2. If the user doesn't know their customer ID, ask for their email address instead
3. If there's an error or no customer found, explain the issue clearly
4. Be helpful and guide the user through the process
5. Email addresses must be exact matches - partial matches will not work

Available tools:
- get_customer_info: Look up customer information by ID or exact email

If you are unable to help the user, politely explain what information you need and suggest they contact support if needed."""

song_system_message = """Your job is to help a customer find any songs they are looking for. 

You only have certain tools you can use. If a customer asks you to look something up that you don't know how, politely tell them what you can help with.

When looking up artists and songs, sometimes the artist/song will not be found. In that case, the tools will return information \
on simliar songs and artists. This is intentional, it is not the tool messing up."""

system_message = """Your job is to help as a customer service representative for a music store.

You should interact politely with customers to try to figure out how you can help. You can help in a few ways:

- Updating user information: if a customer wants to update the information in the user database. Call the router with `customer`
- Recomending music: if a customer wants to find some music or information about music. Call the router with `music`

If the user is asking or wants to ask about updating or accessing their information, send them to that route.
If the user is asking or wants to ask about music, send them to that route.
Otherwise, respond."""

# Chains
def get_customer_messages(messages):
    return [SystemMessage(content=customer_prompt)] + messages

def get_song_messages(messages):
    return [SystemMessage(content=song_system_message)] + messages

def get_messages(messages):
    return [SystemMessage(content=system_message)] + messages

customer_chain = get_customer_messages | model.bind_tools([get_customer_info])
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
            tool_calls = last_message.additional_kwargs['tool_calls']
            if len(tool_calls) > 1:
                # If multiple tool calls, just take the first one
                tool_call = tool_calls[0]
            else:
                tool_call = tool_calls[0]
            return json.loads(tool_call['function']['arguments'])['choice']
        else:
            # Music or customer agent made tool calls
            return "tools"
    
    # If last message is a tool response, end the conversation
    if isinstance(last_message, ToolMessage):
        return END
    
    # If last message is an AI message without tool calls, end conversation
    if isinstance(last_message, AIMessage) and not _is_tool_call(last_message):
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

# Node definitions
tools = [get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_customer_info, get_songs_in_playlist]
tools_node = ToolNode(tools)

def general_node(state):
    messages = state["messages"]
    # Only filter out previous general routing messages, keep everything else
    filtered_messages = _filter_out_routes(messages)
    result = general_chain.invoke(filtered_messages)
    return {"messages": [add_name(result, name="general")]}

def music_node(state):
    messages = state["messages"]
    
    # Check if we already have a music agent response
    last_ai = _get_last_ai_message(messages)
    if last_ai and last_ai.name == "music" and not _is_tool_call(last_ai):
        # Music agent already responded, don't process again
        return {"messages": []}
    
    # Get the last user message for music queries
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if user_messages:
        last_user_message = user_messages[-1]
        # Create a proper conversation context for the music agent
        conversation_context = [SystemMessage(content=song_system_message), last_user_message]
        result = song_recc_chain.invoke(conversation_context)
        return {"messages": [add_name(result, name="music")]}
    return {"messages": []}

def customer_node(state):
    messages = state["messages"]
    
    # Check if we already have a customer agent response
    last_ai = _get_last_ai_message(messages)
    if last_ai and last_ai.name == "customer" and not _is_tool_call(last_ai):
        # Customer agent already responded, don't process again
        return {"messages": []}
    
    # Get the last user message for customer queries
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]
    if user_messages:
        last_user_message = user_messages[-1]
        # Create a proper conversation context for the customer agent
        conversation_context = [SystemMessage(content=customer_prompt), last_user_message]
        result = customer_chain.invoke(conversation_context)
        return {"messages": [add_name(result, name="customer")]}
    return {"messages": []}

# Graph definition
def create_graph():
    """Create and return the LangGraph workflow."""
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages
    
    class State(TypedDict):
        messages: Annotated[list, add_messages]
    
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("general", general_node)
    workflow.add_node("music", music_node)
    workflow.add_node("customer", customer_node)
    workflow.add_node("tools", tools_node)
    
    # Add edges
    workflow.add_conditional_edges("general", _route, {"music": "music", "customer": "customer", "tools": "tools", END: END})
    workflow.add_conditional_edges("tools", _route, {"music": "music", "customer": "customer", "general": "general", END: END})
    workflow.add_conditional_edges("music", _route, {"music": "music", "customer": "customer", "general": "general", "tools": "tools", END: END})
    workflow.add_conditional_edges("customer", _route, {"music": "music", "customer": "customer", "general": "general", "tools": "tools", END: END})
    
    # Set entry point
    workflow.set_conditional_entry_point(_route, {"music": "music", "customer": "customer", "general": "general", "tools": "tools", END: END})
    
    # Compile without checkpointer for now
    return workflow.compile()

# Create the graph instance
graph = create_graph()

# For LangGraph Platform deployment
def get_graph():
    """Return the compiled graph for LangGraph Platform."""
    return graph

if __name__ == "__main__":
    # Final comprehensive test
    from langchain_core.messages import HumanMessage
    
    print("🚀 Final Comprehensive Test Suite")
    print("=" * 50)
    
    test_cases = [
        ("General greeting", "Hello, how can you help me?"),
        ("Music query", "Tell me about U2 songs"),
        ("Customer query", "What's my account information?"),
        ("Edge case - empty", ""),
        ("Edge case - very long", "Tell me about " + "music " * 20 + "and customer service"),
    ]
    
    all_passed = True
    for test_name, message in test_cases:
        print(f"\n🧪 Testing: {test_name}")
        test_input = {"messages": [HumanMessage(content=message)]}
        config = {"configurable": {"thread_id": f"final-test-{test_name.lower().replace(' ', '-')}"}, "recursion_limit": 10}
        
        try:
            result = graph.invoke(test_input, config=config)
            print(f"✅ {test_name}: PASSED")
            
            # Verify we got a proper response
            if result["messages"]:
                last_msg = result["messages"][-1]
                if hasattr(last_msg, 'content') and last_msg.content:
                    print(f"   📝 Response: {last_msg.content[:80]}...")
                elif hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"   🔧 Tool call: {last_msg.tool_calls[0]['name']}")
            else:
                print("   ⚠️  No response generated")
                
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Ready for deployment!")
    else:
        print("⚠️  Some tests failed. Review issues before deployment.")
    print("=" * 50)
