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
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
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

IMPORTANT: Always review the conversation history to understand what the customer has asked about previously. You can reference previous topics, questions, or information they mentioned.

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
- Base your routing decision on the latest user message. Only if the latest message is ambiguous, ask a follow-up question. Do not consider older messages for routing.
- Emit EXACTLY ONE call to the Router tool (choose one of: music OR customer). Never call both.
- Do not route if you can answer directly from history.

When routing to a specialist, include the user's exact request in your assistant message content so the specialist knows what to help with. Format: "User request: [exact user message]"""

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
            # Customer agent made tool calls - route to customer_tools
            return "customer_tools"
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
customer_tools = [get_customer_info]

music_tools_node = ToolNode(music_tools)
customer_tools_node = ToolNode(customer_tools)

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
    
    # Build context
    conversation_context = [SystemMessage(content=system_message)]
    
    # Include recent conversation history for better context awareness
    # Include last 3 human messages and last 5 safe AI messages (from any agent)
    recent_humans = human_messages[-3:] if len(human_messages) > 3 else human_messages
    for hm in recent_humans:
        conversation_context.append(hm)
    
    recent_safe_ai = safe_ai_messages[-5:] if len(safe_ai_messages) > 5 else safe_ai_messages
    for am in recent_safe_ai:
        conversation_context.append(am)
    
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
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    
    # Build context with conversation history
    conversation_context = [SystemMessage(content=song_system_message)]
    
    # Include recent human messages (last 3) for context
    recent_humans = human_messages[-3:] if len(human_messages) > 3 else human_messages
    for hm in recent_humans:
        conversation_context.append(hm)
    
    # Include tool results if available (for formulating responses with data)
    if tool_messages:
        recent_tools = tool_messages[-3:] if len(tool_messages) > 3 else tool_messages
        for tm in recent_tools:
            # Add tool results as system messages for context
            content = tm.content if hasattr(tm, 'content') and isinstance(tm.content, str) else str(tm.content)
            if len(content) > 1000:
                content = content[:1000] + "..."
            tool_name = getattr(tm, "name", "tool")
            conversation_context.append(SystemMessage(content=f"Tool result from {tool_name}: {content}"))
    
    result = song_recc_chain.invoke(conversation_context)
    return {"messages": [add_name(result, name="music")]}

def customer_node(state):
    messages = state["messages"]
    
    # For customer agent, pass recent conversation context for better responses
    # Include recent human messages and safe AI responses (no tool calls)
    human_messages = [m for m in messages if isinstance(m, HumanMessage)]
    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    
    # Filter out AI messages with tool calls to avoid OpenAI errors
    safe_ai_messages = []
    for ai_msg in ai_messages:
        if not (hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls):
            safe_ai_messages.append(ai_msg)
    
    # Build context: recent human messages + safe AI context
    conversation_context = [SystemMessage(content=customer_prompt)]
    
    # Add recent human messages (last 3 for context)
    recent_human = human_messages[-3:] if len(human_messages) > 3 else human_messages
    conversation_context.extend(recent_human)
    
    # Add any recent safe AI messages for context (last 2)
    recent_safe_ai = safe_ai_messages[-2:] if len(safe_ai_messages) > 2 else safe_ai_messages
    conversation_context.extend(recent_safe_ai)
    
    # Include tool results if available (for formulating responses with data)
    if tool_messages:
        recent_tools = tool_messages[-3:] if len(tool_messages) > 3 else tool_messages
        for tm in recent_tools:
            # Add tool results as system messages for context
            content = tm.content if hasattr(tm, 'content') and isinstance(tm.content, str) else str(tm.content)
            if len(content) > 1000:
                content = content[:1000] + "..."
            tool_name = getattr(tm, "name", "tool")
            conversation_context.append(SystemMessage(content=f"Tool result from {tool_name}: {content}"))
    
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
    workflow.add_node("general", general_node)
    workflow.add_node("music", music_node)
    workflow.add_node("customer", customer_node)
    workflow.add_node("music_tools", music_tools_node)
    workflow.add_node("customer_tools", customer_tools_node)
    
    # Add edges with proper routing restrictions
    workflow.add_edge(START, "general") #always start with general
    workflow.add_conditional_edges("general", _route, {"music": "music", "customer": "customer", "general": "general", END: END})
    # Specialist agents can route to their tools, back to themselves, to general, or END
    workflow.add_conditional_edges("music", _route, {"music_tools": "music_tools", "music": "music", "general": "general", END: END})
    workflow.add_conditional_edges("customer", _route, {"customer_tools": "customer_tools", "customer": "customer", "general": "general", END: END})
    # Tools route back to their specialist agent or to general
    workflow.add_conditional_edges("music_tools", _route, {"music": "music", "general": "general"})
    workflow.add_conditional_edges("customer_tools", _route, {"customer": "customer", "general": "general"})
    
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
                config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 15}
                
                result = graph.invoke(graph_input, config=config)
                
                # Get the bot's response
                if result["messages"]:
                    last_msg = result["messages"][-1]
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
