"""
Corporate Intelligence Engine - Streamlit Frontend

An interactive web UI for the corporate-intelligence-engine that connects to the
FastAPI backend and provides real-time visualization of agent reasoning steps.

Features:
  - Chat interface for user queries
  - Real-time agent activity visualization
  - Structured report display
  - Execution time tracking
  - Error handling and recovery

Run with:
  streamlit run frontend.py
"""

import streamlit as st
import requests
import json
from datetime import datetime, timedelta
from typing import Optional
import time

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Corporate Intelligence Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    
    .status-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
    
    .log-entry {
        font-family: monospace;
        font-size: 0.85em;
        line-height: 1.5;
        color: #333;
    }
    
    .success-badge {
        color: #28a745;
        font-weight: bold;
    }
    
    .error-badge {
        color: #dc3545;
        font-weight: bold;
    }
    
    .routing-badge {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 3px;
        font-weight: bold;
        margin: 5px 0;
    }
    
    .research-badge {
        background-color: #e3f2fd;
        color: #1565c0;
    }
    
    .general-badge {
        background-color: #f3e5f5;
        color: #6a1b9a;
    }
    
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_response" not in st.session_state:
    st.session_state.last_response = None

if "api_available" not in st.session_state:
    st.session_state.api_available = None

if "approval_submitted" not in st.session_state:
    st.session_state.approval_submitted = False

if "approval_timeout" not in st.session_state:
    st.session_state.approval_timeout = None

if "pending_approval_request" not in st.session_state:
    st.session_state.pending_approval_request = None

if "trade_completed" not in st.session_state:
    st.session_state.trade_completed = False


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

API_BASE_URL = "http://localhost:8002"
API_HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
API_ANALYZE_ENDPOINT = f"{API_BASE_URL}/api/analyze"
API_STATUS_ENDPOINT = f"{API_BASE_URL}/api/requests"
REQUEST_TIMEOUT = 600  # seconds (10 minutes max - increased for OAuth/MCP calls)
INACTIVITY_TIMEOUT = 180  # seconds (3 minutes of no log updates = stuck)
POLL_INTERVAL = 5  # seconds (poll every 5 seconds for new logs)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_api_health() -> bool:
    """
    Check if the FastAPI backend is running and healthy.
    
    Returns:
        bool: True if API is available, False otherwise
    """
    try:
        response = requests.get(API_HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except Exception as e:
        st.session_state.api_available = False
        return False


def call_analysis_api_with_streaming(user_input: str, log_placeholder) -> Optional[dict]:
    """
    Call the FastAPI backend and stream logs in real-time via polling.
    
    Note: The backend currently blocks on the full response, so polling
    happens AFTER the request completes. For true streaming, the backend
    would need to be async and return request_id immediately.
    
    Args:
        user_input: User query
        log_placeholder: Streamlit placeholder to update logs in real-time
        
    Returns:
        Final response JSON if successful, None otherwise
    """
    try:
        payload = {"user_input": user_input}
        
        st.info("📤 Sending request to backend (this may take 60-120 seconds)...")
        
        # The POST request will block until the entire analysis is complete
        response = requests.post(
            API_ANALYZE_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        
        if response.status_code == 200:
            result = response.json()
            request_id = result.get("request_id")
            
            st.success(f"✓ Request complete! Request ID: {request_id}")
            
            # Display all captured logs from the response
            all_logs = result.get("logs", [])
            
            if all_logs:
                log_text = "\n".join(all_logs)
                with log_placeholder.container():
                    st.code(log_text, language="log")
                st.success(f"✓ {len(all_logs)} log entries captured")
            else:
                st.warning("No logs were captured")
            
            return result
            
        else:
            st.error(f"❌ API Error: {response.status_code}")
            try:
                st.write("Response:", response.json())
            except:
                st.write("Response:", response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ **Backend Connection Error**: Could not reach the FastAPI server. "
            "Please ensure the backend is running:\n\n"
            "```\nC:\\Users\\anand\\anaconda3\\python.exe backend.py\n```"
        )
        return None
    except requests.exceptions.Timeout:
        st.error(
            "❌ **Timeout Error**: The backend took longer than 5 minutes to respond. "
            "\n\nPossible causes:"
            "\n- Qwen LLM is taking too long (typical: 60-120 seconds)"
            "\n- Backend crashed or is stuck"
            "\n- Network connection lost"
            "\n\nCheck the backend terminal for errors."
        )
        return None
    except Exception as e:
        st.error(f"❌ **Error**: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None


def call_analysis_api(user_input: str) -> Optional[dict]:
    """
    Call the FastAPI backend to analyze the user input.
    
    Args:
        user_input: User query
        
    Returns:
        Response JSON if successful, None otherwise
        
    Notes:
        Supports request tracking for live log polling.
        Current implementation waits for full response (blocking).
        
        Architecture:
        1. Creates request with ID on backend
        2. Logs are captured per-request in backend
        3. Frontend polls /api/requests/{request_id}/status for live logs
        4. Response includes request_id for future polling/cancellation
    """
    try:
        payload = {"user_input": user_input}
        
        response = requests.post(
            API_ANALYZE_ENDPOINT,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ **Backend Connection Error**: Could not reach the FastAPI server. "
            "Please ensure the backend is running with `uvicorn backend:app --reload`"
        )
        return None
    except requests.exceptions.Timeout:
        st.error(
            "❌ **Timeout Error**: The backend took too long to respond (5 minutes). "
            "\n\nPossible reasons:"
            "\n- Qwen LLM is generating a detailed response (can take 1-2 minutes)"
            "\n- Backend is stuck on a single operation"
            "\n- Network latency is high"
            "\n\nTip: Check the backend logs to see what operation is taking time."
        )
        return None
    except Exception as e:
        st.error(f"❌ **Error**: {str(e)}")
        return None


def get_routing_badge_class(routing_decision: str) -> str:
    """Get CSS class for routing decision badge."""
    if routing_decision == "research":
        return "research-badge"
    elif routing_decision == "general_q":
        return "general-badge"
    else:
        return ""


def get_routing_badge_label(routing_decision: str) -> str:
    """Get display label for routing decision."""
    if routing_decision == "research":
        return "🔬 Research"
    elif routing_decision == "general_q":
        return "💡 General Question"
    else:
        return "❓ Unknown"


# ============================================================================
# PAGE HEADER
# ============================================================================

st.markdown(
    "<h1 class='main-title'>📊 Corporate Intelligence & Earnings Analyst Engine</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style='text-align: center; color: #666; margin-bottom: 20px;'>
    Powered by Multi-Agent AI Orchestration | Real-time Financial Intelligence
    </p>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# SIDEBAR - STATUS & HELP
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # API Health Status
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**Backend Status**")
    with col2:
        if check_api_health():
            st.markdown("<span style='color: #28a745; font-weight: bold;'>✓ Online</span>", unsafe_allow_html=True)
            st.session_state.api_available = True
        else:
            st.markdown("<span style='color: #dc3545; font-weight: bold;'>✗ Offline</span>", unsafe_allow_html=True)
            st.session_state.api_available = False
    
    if not st.session_state.api_available:
        st.warning(
            "⚠️ **Backend not running!**\n\n"
            "Start the FastAPI server in another terminal:\n\n"
            "```bash\n"
            "uvicorn backend:app --reload\n"
            "```"
        )
    
    st.divider()
    
    st.markdown("### 📖 Help")
    st.markdown("""
    **What can I ask?**
    
    **Research Queries:**
    - "Analyze NVDA earnings"
    - "What's the current price of TSLA?"
    - "Show me Tesla stock forecast"
    
    **General Questions:**
    - "Top ML frameworks 2024?"
    - "What is quantitative analysis?"
    - "Explain financial derivatives"
    
    **How it works:**
    1. Enter your query
    2. Agent triages request (research vs general)
    3. Routes to appropriate analyzer
    4. Generates structured report
    """)
    
    st.divider()
    
    st.markdown("### 🔗 Links")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("[📚 Docs](http://localhost:8000/docs)")
    with col2:
        st.markdown("[🔄 Reload](javascript:location.reload())")
    with col3:
        st.markdown("[ℹ️ About](#)")


# ============================================================================
# MAIN CONTENT - CHAT INTERFACE
# ============================================================================

st.markdown("### 💬 Analysis Interface")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display metadata if it's an assistant response
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Routing Decision",
                    get_routing_badge_label(metadata.get("routing_decision", "Unknown")),
                )
            
            with col2:
                st.metric(
                    "Execution Time",
                    f"{metadata.get('execution_time_ms', 0):.1f}ms"
                )
            
            with col3:
                st.metric(
                    "Log Entries",
                    len(metadata.get("logs", []))
                )


# ============================================================================
# USER INPUT & PROCESSING
# ============================================================================

st.markdown("---")

# Chat input box
user_input = st.chat_input(
    "Enter a ticker or analysis prompt (e.g., Analyze NVDA)...",
    key="user_input",
)

if user_input:
    # Reset trade completion flag for new requests
    st.session_state.trade_completed = False
    
    # Check API availability first
    if not check_api_health():
        st.error(
            "❌ **Cannot process request**: Backend is not running. "
            "Please start the FastAPI server first."
        )
    else:
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Create status container for real-time updates
        with st.status("🚀 Orchestrating AI Agents...", expanded=True) as status:
            st.write("Initiating multi-agent analysis workflow...")
            
            # Create placeholder for live logs
            log_placeholder = st.empty()
            
            # Call the API with streaming logs
            response = call_analysis_api_with_streaming(user_input, log_placeholder)
            
            # AGGRESSIVE DEBUG
            st.write("---")
            st.write("### 🚨 AGGRESSIVE DEBUG OUTPUT")
            st.write(f"✓ Response object type: `{type(response)}`")
            st.write(f"✓ Response is None: `{response is None}`")
            if response:
                st.write(f"✓ Response status: `{response.get('status')}`")
                st.write(f"✓ Response keys: `{list(response.keys())}`")
                st.write(f"✓ pending_approval: `{response.get('pending_approval')}`")
                st.write(f"✓ request_id: `{response.get('request_id')}`")
            st.write("---")
            
            if response:
                st.write(f"**[DEBUG] Response Status:** `{response.get('status')}`")
                st.write(f"**[DEBUG] Pending Approval:** `{response.get('pending_approval')}`")
                
                # Display logs from response
                logs = response.get("logs", [])
                if logs:
                    st.write("---")
                    st.write("### 📋 Captured Agent Logs")
                    log_text = "\n".join(logs)
                    st.code(log_text, language="log")
                else:
                    st.warning("⚠️ No logs captured in response")
                
                # Check for errors
                if response.get("status") == "error":
                    status.update(
                        label="❌ Analysis Failed!",
                        state="error",
                        expanded=False,
                    )
                    st.error(f"Error: {response.get('error_message', 'Unknown error')}")
                
                # Check for approval pending
                elif response.get("status") == "awaiting_approval":
                    st.info("✅ Status check passed: awaiting_approval detected")
                    status.update(
                        label="⏳ Awaiting Human Approval",
                        state="running",
                        expanded=True,
                    )
                    
                    # Store approval request in session state so it persists after container closes
                    pending_approval = response.get("pending_approval", {})
                    request_id = response.get("request_id", "")
                    st.session_state.pending_approval_request = {
                        "request_id": request_id,
                        "pending_approval": pending_approval,
                        "response": response,
                    }
                    
                    st.warning("🔒 **TRADE APPROVAL REQUIRED** (⏱️ 10 second timeout)")
                    
                    # Display pending approval details
                    pending_approval = response.get("pending_approval", {})
                    request_id = response.get("request_id", "")
                    
                    if pending_approval:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Trade Details:**")
                            st.info(
                                f"**Action:** {pending_approval.get('action', 'N/A')}\n\n"
                                f"**Ticker:** {pending_approval.get('ticker', 'N/A')}\n\n"
                                f"**Amount:** {pending_approval.get('amount', 'N/A')}\n\n"
                                f"**Request ID:** `{request_id}`"
                            )
                        
                        with col2:
                            st.markdown("**Reasoning:**")
                            st.text(
                                pending_approval.get('reasoning', 'No reasoning provided')
                            )
                    
                    # Approval buttons - NO BLOCKING TIMER
                    st.markdown("---")
                    st.markdown("### ✅ Approval Decision")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Approve Trade", key=f"approve_{request_id}", use_container_width=True):
                            with st.spinner("Submitting approval and executing trade..."):
                                try:
                                    # Step 1: Submit approval
                                    approve_response = requests.post(
                                        f"{API_BASE_URL}/api/approve/{request_id}",
                                        json={"approved": True, "approver_notes": "Approved via Streamlit UI"},
                                        timeout=120  # Increased for OAuth/MCP
                                    )
                                    
                                    if approve_response.status_code != 200:
                                        error_resp = approve_response.json()
                                        st.error(f"❌ Failed to submit approval: {approve_response.status_code}")
                                        st.error(
                                            f"**Error:** {error_resp.get('detail', 'Unknown error')}\n\n"
                                            f"**Status Code:** {approve_response.status_code}"
                                        )
                                    else:
                                        st.success("✅ Approval stored!")
                                        
                                        # Step 2: Execute the trade
                                        time.sleep(0.5)
                                        execute_response = requests.post(
                                            f"{API_BASE_URL}/api/execute/{request_id}",
                                            timeout=120  # Increased for OAuth/MCP
                                        )
                                        
                                        if execute_response.status_code == 200:
                                            result = execute_response.json()
                                            st.success("✅ **TRADE EXECUTED SUCCESSFULLY!**")
                                            
                                            # Format and display the trade execution result
                                            trade_details = result.get("trade_details", {})
                                            execution_result = result.get("execution_result", {})
                                            
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.markdown("### 📋 Trade Details")
                                                st.info(
                                                    f"**Ticker:** {trade_details.get('ticker', 'N/A')}\n\n"
                                                    f"**Action:** {trade_details.get('action', 'N/A')}\n\n"
                                                    f"**Amount:** {trade_details.get('amount', 'N/A')}\n\n"
                                                    f"**Quantity:** {trade_details.get('quantity', 'N/A')}"
                                                )
                                            
                                            with col2:
                                                st.markdown("### 📊 Order Execution")
                                                filled_price = execution_result.get('filled_price')
                                                quantity = execution_result.get('quantity')
                                                st.success(
                                                    f"**Order ID:** `{execution_result.get('order_id', 'N/A')}`\n\n"
                                                    f"**Status:** {execution_result.get('status', 'N/A').upper()}\n\n"
                                                    f"**Fill Price:** ${filled_price:.2f}\n\n"
                                                    f"**Quantity Filled:** {quantity:.4f} shares"
                                                )
                                            
                                            st.markdown("---")
                                            st.markdown("### ✅ Summary")
                                            st.success(
                                                f"💰 **Total Value:** ${execution_result.get('total_value', 0):.2f}\n\n"
                                                f"📅 **Filled At:** {execution_result.get('filled_at', 'N/A')}\n\n"
                                                f"🎮 **Simulated Trade:** {'Yes' if execution_result.get('simulated') else 'No'}"
                                            )
                                        else:
                                            error_resp = execute_response.json()
                                            st.error(f"❌ Execution failed: {execute_response.status_code}")
                                            st.error(
                                                f"**Error:** {error_resp.get('detail', 'Unknown error')}\n\n"
                                                f"**Status Code:** {execute_response.status_code}"
                                            )
                                except Exception as e:
                                    st.error(f"❌ Error: {str(e)}")
                                    import traceback
                                    st.write(traceback.format_exc())
                    
                    with col2:
                        if st.button("❌ Reject Trade", key=f"reject_{request_id}", use_container_width=True):
                            with st.spinner("Submitting rejection..."):
                                try:
                                    reject_response = requests.post(
                                        f"{API_BASE_URL}/api/approve/{request_id}",
                                        json={"approved": False, "approver_notes": "Rejected via Streamlit UI"},
                                        timeout=120  # Increased for OAuth/MCP
                                    )
                                    
                                    if reject_response.status_code == 200:
                                        st.error("❌ **TRADE REJECTED**")
                                    else:
                                        error_resp = reject_response.json()
                                        st.error(f"Failed to submit rejection: {reject_response.status_code}")
                                        st.error(
                                            f"**Error:** {error_resp.get('detail', 'Unknown error')}\n\n"
                                            f"**Status Code:** {reject_response.status_code}"
                                        )
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")


                
                else:
                    # Update status to complete
                    status.update(
                        label=f"✅ Analysis Complete! ({response.get('execution_time_ms', 0):.1f}ms)",
                        state="complete",
                        expanded=False,
                    )
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.get("report_markdown", ""),
                        "metadata": {
                            "routing_decision": response.get("routing_decision", ""),
                            "logs": response.get("logs", []),
                            "execution_time_ms": response.get("execution_time_ms", 0),
                        }
                    })
                    
                    # Display final report
                    st.markdown("---")
                    st.markdown("### 📊 Final Report")
                    
                    st.markdown(
                        f"<div class='routing-badge {get_routing_badge_class(response.get('routing_decision'))}'>"
                        f"{get_routing_badge_label(response.get('routing_decision'))}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    
                    st.markdown(response.get("report_markdown", ""))
            else:
                status.update(
                    label="❌ Request Failed",
                    state="error",
                    expanded=False,
                )


# =========================================================================
# APPROVAL UI - OUTSIDE THE STATUS CONTAINER (PERSISTS AFTER CONTAINER CLOSES)
# =========================================================================

# Only show approval UI if trade hasn't been completed
if st.session_state.get("pending_approval_request") and not st.session_state.get("trade_completed"):
    approval_data = st.session_state.pending_approval_request
    request_id = approval_data.get("request_id", "")
    pending_approval = approval_data.get("pending_approval", {})
    
    st.markdown("---")
    st.warning("🔒 **TRADE APPROVAL REQUIRED** - Click buttons below")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Trade Details:**")
        st.info(
            f"**Action:** {pending_approval.get('action', 'N/A')}\n\n"
            f"**Ticker:** {pending_approval.get('ticker', 'N/A')}\n\n"
            f"**Amount:** {pending_approval.get('amount', 'N/A')}\n\n"
            f"**Request ID:** `{request_id}`"
        )
    
    with col2:
        st.markdown("**Reasoning:**")
        st.text(
            pending_approval.get('reasoning', 'No reasoning provided')
        )
    
    # Approval buttons
    st.markdown("---")
    st.markdown("### ✅ Approval Decision")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Approve Trade", key=f"approve_{request_id}_persistent", use_container_width=True):
            with st.spinner("Submitting approval and executing trade..."):
                try:
                    # Step 1: Submit approval
                    approve_response = requests.post(
                        f"{API_BASE_URL}/api/approve/{request_id}",
                        json={"approved": True, "approver_notes": "Approved via Streamlit UI"},
                        timeout=120  # Increased for OAuth/MCP
                    )
                    
                    if approve_response.status_code != 200:
                        error_resp = approve_response.json()
                        st.error(f"❌ Failed to submit approval: {approve_response.status_code}")
                        st.error(
                            f"**Error:** {error_resp.get('detail', 'Unknown error')}\n\n"
                            f"**Status Code:** {approve_response.status_code}"
                        )
                    else:
                        st.success("✅ Approval stored!")
                        
                        # Step 2: Execute the trade
                        time.sleep(0.5)
                        execute_response = requests.post(
                            f"{API_BASE_URL}/api/execute/{request_id}",
                            timeout=120  # Increased for OAuth/MCP
                        )
                        
                        if execute_response.status_code == 200:
                            result = execute_response.json()
                            st.success("✅ **TRADE EXECUTED SUCCESSFULLY!**")
                            
                            # Format and display the trade execution result
                            trade_details = result.get("trade_details", {})
                            execution_result = result.get("execution_result", {})
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### 📋 Trade Details")
                                st.info(
                                    f"**Ticker:** {trade_details.get('ticker', 'N/A')}\n\n"
                                    f"**Action:** {trade_details.get('action', 'N/A')}\n\n"
                                    f"**Amount:** {trade_details.get('amount', 'N/A')}\n\n"
                                    f"**Quantity:** {trade_details.get('quantity', 'N/A')}"
                                )
                            
                            with col2:
                                st.markdown("### 📊 Order Execution")
                                filled_price = execution_result.get('filled_price')
                                quantity = execution_result.get('quantity')
                                st.success(
                                    f"**Order ID:** `{execution_result.get('order_id', 'N/A')}`\n\n"
                                    f"**Status:** {execution_result.get('status', 'N/A').upper()}\n\n"
                                    f"**Fill Price:** ${filled_price:.2f}\n\n"
                                    f"**Quantity Filled:** {quantity:.4f} shares"
                                )
                            
                            st.markdown("---")
                            st.markdown("### ✅ Summary")
                            st.success(
                                f"💰 **Total Value:** ${execution_result.get('total_value', 0):.2f}\n\n"
                                f"📅 **Filled At:** {execution_result.get('filled_at', 'N/A')}\n\n"
                                f"🎮 **Simulated Trade:** {'Yes' if execution_result.get('simulated') else 'No'}"
                            )
                            
                            # Clear the approval request (but don't rerun yet)
                            st.session_state.pending_approval_request = None
                            st.session_state.trade_completed = True
                        else:
                            error_resp = execute_response.json()
                            st.error(f"❌ Execution failed: {execute_response.status_code}")
                            st.error(
                                f"**Error:** {error_resp.get('detail', 'Unknown error')}\n\n"
                                f"**Status Code:** {execute_response.status_code}"
                            )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    st.write(traceback.format_exc())
    
    with col2:
        if st.button("❌ Reject Trade", key=f"reject_{request_id}_persistent", use_container_width=True):
            with st.spinner("Submitting rejection..."):
                try:
                    reject_response = requests.post(
                        f"{API_BASE_URL}/api/approve/{request_id}",
                        json={"approved": False, "approver_notes": "Rejected via Streamlit UI"},
                        timeout=120  # Increased for OAuth/MCP
                    )
                    
                    if reject_response.status_code == 200:
                        st.error("❌ **TRADE REJECTED**")
                        
                        # Clear the approval request (but don't rerun yet)
                        st.session_state.pending_approval_request = None
                        st.session_state.trade_completed = True
                    else:
                        error_resp = reject_response.json()
                        st.error(f"Failed to submit rejection: {reject_response.status_code}")
                        st.error(
                            f"**Error:** {error_resp.get('detail', 'Unknown error')}\n\n"
                            f"**Status Code:** {reject_response.status_code}"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")


# ============================================================================
# FOOTER & INFO
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.info("💻 **Frontend**: Streamlit")

with col2:
    st.info("⚙️ **Backend**: FastAPI + Uvicorn")

with col3:
    st.info("🧠 **Orchestration**: LangGraph State Machine")

st.markdown("""
---
<p style='text-align: center; color: #999; font-size: 0.9em;'>
Corporate Intelligence Engine | AI State Graph | Financial Research
</p>
""", unsafe_allow_html=True)
