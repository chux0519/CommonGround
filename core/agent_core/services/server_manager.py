import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import mcp.client.session as mcp_session
from fastapi import FastAPI
from mcp.client.session_group import ClientSessionGroup, StdioServerParameters, StreamableHttpParameters
from .file_monitor import start_file_monitoring
from ..iic.core.iic_handlers import BASE_DIR
from ..config.app_config import NATIVE_MCP_SERVERS
from ..framework.tool_registry import initialize_registry
from ..a2a_bridge.router import A2A_ROUTER
from ..a2a_bridge.executors import SmartRAG_A2A_Executor 

logger = logging.getLogger(__name__)

# --- Global variables ---
# --- START: Modified code - Completely remove global SESSION_GROUP ---
# SESSION_GROUP: Optional[ClientSessionGroup] = None
# --- END: Modified code ---
MCP_SESSION_POOL: Optional[asyncio.Queue] = None

# --- START: Modified code - Remove initialize_session_group_and_tools and shutdown_session_group ---
# async def initialize_session_group_and_tools(): ...
# async def shutdown_session_group(): ...
# --- END: Modified code ---

@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    """
    [REFACTORED] FastAPI's lifespan manager.
    Initializes all application-level resources including Logging, MCP, Tool Registry,
    the new A2A In-Memory Bridge, and File Monitoring.
    """
    # ==================================================================
    # PHASE 1: PRE-INITIALIZATION (Logging)
    # This part remains unchanged.
    # ==================================================================
    # ... (your existing logging reconfiguration code) ...
    # Re-configure logging after uvicorn potentially modified it
    try:
        from ..config.logging_config import setup_global_logging
        import os
        log_level = os.getenv("LOG_LEVEL", "INFO")
        log_file = os.getenv("LOG_FILE", None)
        setup_global_logging(log_level, log_file)
        
        # Additional suppression for MCP-related warnings
        import logging
        logging.getLogger("mcp").setLevel(logging.ERROR)
        logging.getLogger("mcp.client").setLevel(logging.ERROR)
        logging.getLogger("mcp.server").setLevel(logging.ERROR)
        
        logger.info("logging_reconfigured_in_lifespan", extra={"description": "Logging reconfigured in lifespan manager"})
    except Exception as e:
        logger.error("logging_reconfiguration_failed", extra={"description": "Failed to reconfigure logging", "error": str(e)})
    
    logger.info("application_startup_begin", extra={"description": "Initializing resources via lifespan manager"})

    # ==================================================================
    # PHASE 2: TOOL & AGENT INITIALIZATION
    # This is the main area of change.
    # ==================================================================

    # 2.1: Initialize MCP Session Pool (remains unchanged)
    global MCP_SESSION_POOL
    MCP_SESSION_POOL = asyncio.Queue()
    logger.info("mcp_session_pool_initialized")
    
    # 2.2: Discover external tools via MCP (remains unchanged)
    logger.info("tool_discovery_session_create_begin")
    discovery_session_group = await initialize_mcp_session_for_context()
    
    # 2.3: Initialize the Tool Registry (remains unchanged)
    # This step is crucial. It loads all your Python-based tools (`@tool_registry`)
    # AND any discovered MCP tools into the central `_TOOL_REGISTRY`.
    if discovery_session_group:
        await initialize_registry(discovery_session_group, "agent_core/nodes/custom_nodes")
        await release_mcp_session_to_pool(discovery_session_group)
        logger.info("tool_discovery_session_pooled")
    else:
        logger.warning("tool_discovery_session_failed")
        # Still initialize with internal tools even if MCP fails
        await initialize_registry(None, "agent_core/nodes/custom_nodes")

    # +++ 2.4: NEW - Initialize and Register In-Memory A2A Agents +++
    # This new block runs AFTER the tool registry is initialized, because our
    # A2A Executors might depend on it (e.g., to find tool node classes).
    try:
        logger.info("a2a_bridge_registration_starting")
        
        # Instantiate and register each built-in Associate's executor
        A2A_ROUTER.register_agent("A2A_SmartRAG", SmartRAG_A2A_Executor())
        # As you create more executors, register them here:
        # A2A_ROUTER.register_agent("A2A_WebDeveloper", WebDeveloperA2AExecutor())
        
        logger.info("a2a_bridge_registration_complete", extra={"registered_agents": list(A2A_ROUTER._executors.keys())})
    except Exception as e:
        logger.error("a2a_bridge_registration_failed", exc_info=True)
    # +++ END OF NEW BLOCK +++

    # ==================================================================
    # PHASE 3: BACKGROUND SERVICES INITIALIZATION
    # ==================================================================

    # 3.1: Start File Monitor (remains unchanged)
    logger.info("file_monitor_start", extra={"directory": BASE_DIR})
    start_file_monitoring(BASE_DIR, loop=asyncio.get_running_loop())
    
    # Let the application run
    logger.info("application_startup_complete")
    yield
    
    # ==================================================================
    # PHASE 4: SHUTDOWN & CLEANUP
    # This part remains unchanged.
    # ==================================================================
    logger.info("application_shutdown_begin")
    
    # ... (your existing MCP pool cleanup code) ...
    if MCP_SESSION_POOL:
        logger.info("mcp_session_pool_cleanup_begin", extra={"session_count": MCP_SESSION_POOL.qsize()})
        while not MCP_SESSION_POOL.empty():
            try:
                session_to_close = MCP_SESSION_POOL.get_nowait()
                try:
                    await session_to_close.__aexit__(None, None, None)
                except Exception:
                    logger.warning("mcp_session_graceful_exit_failed")
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error("mcp_session_pool_shutdown_error", exc_info=True)
        logger.info("mcp_session_pool_cleanup_complete")

    logger.info("application_shutdown_complete")

# --- START: Modified code - Remove get_session_group ---
# def get_session_group() -> Optional[ClientSessionGroup]:
#     return SESSION_GROUP
# --- END: Modified code ---

async def initialize_mcp_session_for_context() -> Optional[ClientSessionGroup]:
    """
    Creates a new ClientSessionGroup instance for a single context and connects to all servers.
    Returns a connected ClientSessionGroup instance, or None on failure.
    """
    logger.info("mcp_session_group_init_begin", extra={"description": "Initializing a new ClientSessionGroup for a specific agent context"})
    
    session_group = ClientSessionGroup()
    
    server_connections = []
    for name, conf in NATIVE_MCP_SERVERS.items():
        transport = conf.get("transport")
        params = None
        if transport == "stdio":
            params = StdioServerParameters(command=conf["command"], args=conf["args"])
        elif transport == "http":
            params = StreamableHttpParameters(url=conf["url"])
        else:
            logger.warning("mcp_unsupported_transport", extra={"description": "Unsupported transport type for server", "transport": transport, "server_name": name})
            continue
        server_connections.append({"name": name, "params": params})

    connect_tasks = [session_group.connect_to_server(conn["params"]) for conn in server_connections]
    results = await asyncio.gather(*connect_tasks, return_exceptions=True)

    for i, result_or_exc in enumerate(results):
        server_name = server_connections[i]["name"]
        if isinstance(result_or_exc, Exception):
            logger.error("mcp_server_connection_failed", extra={"description": "Context-specific MCP connection failed", "server_name": server_name, "error": str(result_or_exc)}, exc_info=result_or_exc)
            return None
        else:
            result_or_exc.server_name_from_config = server_name
            logger.info("mcp_server_connection_success", extra={"description": "Context-specific MCP connection successful", "server_name": server_name})
            
    return session_group

# --- START: New code - Session pool management functions ---
async def acquire_mcp_session_from_pool(max_retries=3) -> Optional[ClientSessionGroup]: # Add retries in case the pool is full of bad connections
    """
    (Modified) Acquires a healthy MCP session group from the global pool.
    Performs a health check before returning the session.
    """
    if MCP_SESSION_POOL is None:
        logger.error("mcp_session_pool_not_initialized", extra={"description": "MCP Session Pool is not initialized"})
        return None

    for attempt in range(max_retries):
        try:
            # 1. Try to get a session from the pool
            session_group = MCP_SESSION_POOL.get_nowait()
            logger.info("mcp_session_acquired_from_pool", extra={"description": "Acquired a session from pool, performing health check", "pool_size": MCP_SESSION_POOL.qsize()})

            # 2. Perform a health check on each session
            is_healthy = True
            if session_group.sessions: # Ensure the session_group contains sessions
                # We only need to ping one session to confirm if the underlying process for the whole group is alive
                # Usually, one group corresponds to one process
                try:
                    # Use asyncio.wait_for with a short timeout to prevent ping from blocking
                    await asyncio.wait_for(session_group.sessions[0].send_ping(), timeout=3.0)
                    logger.info("mcp_session_health_check_passed", extra={"description": "Health check PASSED. Returning session to caller"})
                    return session_group # Healthy, return the session
                except Exception as e:
                    logger.warning("mcp_session_health_check_failed", extra={"description": "Health check FAILED for a pooled session. Discarding it", "error": str(e)})
                    is_healthy = False
                    # Make sure to close this invalid session group to release resources
                    try:
                        await session_group.__aexit__(None, None, None)
                    except Exception:
                        logger.warning("mcp_session_cleanup_limitation", extra={"description": "DUE to MCP SDK Limitation - this session can not gracefully exit. Accumulating too many this error might drain your resource, but it should be fine as far as you are not running this as a service"})
            
            if not is_healthy:
                # If not healthy, continue the loop to try and get the next one
                continue

        except asyncio.QueueEmpty:
            # 3. If the pool is empty, create a new session
            logger.info("mcp_session_pool_empty_creating_new", extra={"description": "MCP session pool is empty. Creating a new session"})
            return await initialize_mcp_session_for_context()

    # If a healthy session cannot be found after multiple loops
    logger.error("mcp_session_acquire_failed_creating_fallback", extra={"description": "Failed to acquire a healthy MCP session after retries. Creating a new one as a last resort", "max_retries": max_retries})
    return await initialize_mcp_session_for_context()

async def release_mcp_session_to_pool(session_group: ClientSessionGroup):
    """
    Releases a used MCP session group back to the global pool.
    """
    if MCP_SESSION_POOL is None:
        logger.error("mcp_session_pool_not_initialized_release", extra={"description": "MCP Session Pool is not initialized! Cannot release session"})
        return
    if session_group:
        await MCP_SESSION_POOL.put(session_group)
        logger.info("mcp_session_released_to_pool", extra={"description": "MCP session released back to the pool", "pool_size": MCP_SESSION_POOL.qsize()})
# --- END: New code ---
