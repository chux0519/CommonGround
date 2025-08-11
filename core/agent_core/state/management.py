# state.py

import logging
import asyncio
import copy
from datetime import datetime, timezone
from typing import Dict, Optional, Any

# Import project dependencies
from agent_profiles.loader import AGENT_PROFILES, SHARED_LLM_CONFIGS
from ..framework.turn_manager import TurnManager
from ..framework.profile_utils import get_active_profile_instance_id_by_name
from ..utils.knowledge_base import KnowledgeBase
from ..models.context import (
    RunContext,
    SubContext,
    SubContextState,
    SubContextRuntimeObjects
)


# --- Module-level Logger ---
logger = logging.getLogger(__name__)

# --- Core Implementation ---

def _create_flow_specific_state_template() -> SubContextState:
    """
    Creates a basic template for a flow's private, serializable state dictionary ('state').
    This function no longer handles identity information (like run_id, agent_id) and focuses only on the pure state.
    """
    return {
        "messages": [],
        "current_action": None,
        "inbox": [],
        "flags": {},
        "deliverables": {},
        "last_activity_timestamp": datetime.now(timezone.utc).isoformat(),
    }

def create_run_context(
    server_run_id: str,
    run_type: str,
    initial_params: Dict,
    event_manager_for_websocket: Any, # SessionEventManager
    project_id: str = "",
) -> RunContext:
    """
    Creates and initializes the globally unique root context object ('RunContext').
    This is the single source of truth for all state and configuration.
    """
    logger.info("run_context_creating", extra={"server_run_id": server_run_id, "run_type": run_type})

    # 1. Assemble run_context
    run_context: RunContext = {
        "meta": {
            "run_id": server_run_id,
            "run_type": run_type,
            "creation_timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "CREATED",
        },
        "config": {
            # Create a deep copy snapshot of the configuration to ensure it's fixed for this run
            "agent_profiles_store": copy.deepcopy(AGENT_PROFILES),
            "shared_llm_configs_ref": copy.deepcopy(SHARED_LLM_CONFIGS),
        },
        "team_state": {
            "question": initial_params.get("question"),
            "work_modules": {},
            "_work_module_next_id": 1,
            "profiles_list_instance_ids": [],
            "is_principal_flow_running": False,
            "dispatch_history": [],
            "turns": [],
            "partner_directives_queue": [],
        },
        "runtime": {
            "event_manager": event_manager_for_websocket,
            "knowledge_base": KnowledgeBase(server_run_id),
            "turn_manager": TurnManager(),
            # --- New fields ---
            "token_usage_stats": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "max_context_window": 0, # Max prompt_tokens + completion_tokens in a single call
                "total_successful_calls": 0,
                "total_failed_calls": 0, # Active failures, i.e., failures after the stream has started
            }
            # --- End of new fields ---
        },
        "sub_context_refs": {
            "_partner_context_ref": None,
            "_principal_context_ref": None,
            "_ongoing_associate_tasks": {},
        },
        "project_id": project_id,
    }

    # 2. Create and associate initial sub-contexts based on the run type
    if run_type == "partner_interaction":
        # The Partner's initial question should also be set in the team_state
        if run_context["team_state"]["question"] is None:
            run_context["team_state"]["question"] = initial_params.get("initial_user_query")
        
        partner_ctx = create_partner_context(run_context_ref=run_context, parent_agent_id=None)
        run_context["sub_context_refs"]["_partner_context_ref"] = partner_ctx
        logger.debug("partner_context_created", extra={"server_run_id": server_run_id})

    elif run_type == "principal_direct":
        # The list of Associate Profiles available to the Principal is provided by the CLI or other callers in initial_params
        cli_profile_names = initial_params.get("cli_mode_default_profiles", [])
        resolved_ids = []
        for name in cli_profile_names:
            instance_id = get_active_profile_instance_id_by_name(run_context["config"]["agent_profiles_store"], name)
            if instance_id:
                resolved_ids.append(instance_id)
        run_context["team_state"]["profiles_list_instance_ids"] = resolved_ids

        principal_ctx = create_principal_context(run_context_ref=run_context, parent_agent_id=None, iteration_mode="start_fresh")
        run_context["sub_context_refs"]["_principal_context_ref"] = principal_ctx
        logger.debug("principal_context_created", extra={"server_run_id": server_run_id})

    # For simple services like chat_completion and fim, their state can exist directly in run_context["state"],
    # but for consistency, we can also create sub-contexts for them or operate directly in the message_handler.
    # The current approach favors handling the state of simple services in the message_handler.

    return run_context

def create_partner_context(run_context_ref: RunContext, parent_agent_id: Optional[str]) -> SubContext:
    """Creates the sub-context for the Partner Agent."""
    agent_id = "Partner"
    
    # 1. Create private state
    partner_state = _create_flow_specific_state_template()

    # 2. Populate Partner-specific runtime objects
    partner_runtime_objects: SubContextRuntimeObjects = {
        "new_user_input_event": asyncio.Event()
    }

    # 3. Assemble Partner context
    partner_context: SubContext = {
        "meta": {
            "run_id": run_context_ref["meta"]["run_id"],
            "agent_id": agent_id,
            "parent_agent_id": parent_agent_id,
        },
        "state": partner_state,
        "runtime_objects": partner_runtime_objects,
        "refs": {
            "run": run_context_ref,
            "team": run_context_ref["team_state"],
        }
    }
    
    # 4. Populate Partner-specific initial state
    # The Partner needs to know all available Profiles to discuss with the user
    all_profiles = run_context_ref["config"]["agent_profiles_store"]
    # Filter based on the available_for_staffing flag
    staffing_available_instance_ids = [
        inst_id for inst_id, prof in all_profiles.items() 
        if prof.get("is_active") and not prof.get("is_deleted") and prof.get("available_for_staffing") is True
    ]
    partner_state["profiles_list_instance_ids"] = staffing_available_instance_ids
    
    # Add the initial question to the message history to provide context for the user's conversation
    initial_query = run_context_ref["team_state"].get("question")
    if initial_query:
        partner_state["messages"].append({"role": "user", "content": initial_query})

    return partner_context

def create_principal_context(
    run_context_ref: RunContext, 
    parent_agent_id: Optional[str],
    iteration_mode: str
) -> SubContext:
    """Creates the sub-context for the Principal Agent."""
    agent_id = "Principal"
    
    principal_state = _create_flow_specific_state_template()
    principal_state["current_iteration_count"] = 1

    principal_context: SubContext = {
        "meta": {
            "run_id": run_context_ref["meta"]["run_id"],
            "agent_id": agent_id,
            "parent_agent_id": parent_agent_id,
            "iteration_mode": iteration_mode,
        },
        "state": principal_state,
        "runtime_objects": {},
        "refs": {
            "run": run_context_ref,
            "team": run_context_ref["team_state"],
        }
    }
    
    # Initial query is now handled via Handover Protocol and inbox, not direct injection.

    return principal_context

def validate_context_state(context: SubContext) -> bool:
    """
    Validates if the structure of a sub-context object is basically correct.
    This is a sanity check, not strict type validation.
    """
    if not isinstance(context, dict): return False
    
    required_top_keys = {"meta", "state", "runtime_objects", "refs"}
    if not required_top_keys.issubset(context.keys()):
        logger.warning("context_validation_failed_missing_keys", extra={"found_keys": list(context.keys())})
        return False

    if not isinstance(context["refs"], dict) or "run" not in context["refs"] or "team" not in context["refs"]:
        logger.warning("context_validation_failed_refs_malformed")
        return False
        
    if not isinstance(context["state"], dict):
        logger.warning("context_validation_failed_state_not_dict")
        return False

    return True

def update_context_activity(context: SubContext):
    """Updates the last activity timestamp of the sub-context."""
    if context and isinstance(context.get("state"), dict):
        context["state"]["last_activity_timestamp"] = datetime.now(timezone.utc).isoformat()
    else:
        agent_id = context.get("meta", {}).get("agent_id", "Unknown")
        logger.warning("context_activity_update_failed", extra={"agent_id": agent_id})

def _inject_restored_state(target_context: RunContext, restored_data: Dict):
    """
    (Refactored for a safer approach)
    Safely injects deserialized state data, loaded from a file, into a brand new,
    fully functional RunContext object.
    """
    if not isinstance(restored_data, dict):
        logger.error("restored_data_not_dict")
        return

    # 1. Inject team_state and KnowledgeBase (logic unchanged)
    if "team_state" in restored_data and isinstance(restored_data["team_state"], dict):
        target_context["team_state"].update(copy.deepcopy(restored_data["team_state"]))

    restored_kb_data = restored_data.get("knowledge_base")
    if restored_kb_data and isinstance(restored_kb_data, dict):
        target_context["runtime"]["knowledge_base"] = KnowledgeBase.from_dict(restored_kb_data)
        logger.info("knowledge_base_restored", extra={"run_id": target_context['meta']['run_id']})
    else:
        if "knowledge_base" not in target_context.get("runtime", {}):
            target_context["runtime"]["knowledge_base"] = KnowledgeBase(target_context['meta']['run_id'])
        logger.warning("knowledge_base_not_found_using_new", extra={"run_id": target_context['meta']['run_id']})

    # 2. Inject the state of each sub-context (new logic)
    restored_sub_states = restored_data.get("sub_contexts_state", {})
    if isinstance(restored_sub_states, dict):
        for context_key, state_data in restored_sub_states.items():
            
            # Check if the target reference already exists (e.g., Partner Context is pre-created)
            target_sub_context_ref = target_context["sub_context_refs"].get(context_key)

            if target_sub_context_ref:
                # If the reference exists, inject the state directly (e.g., Partner)
                if isinstance(state_data, dict):
                    target_sub_context_ref["state"] = copy.deepcopy(state_data)
                    logger.info("sub_context_state_injected", extra={"context_key": context_key})
                else:
                    logger.warning("sub_context_state_not_dict", extra={"context_key": context_key})
            
            elif isinstance(state_data, dict):
                # If the reference does not exist (is None), rebuild the entire SubContext object on-demand
                logger.info("sub_context_rebuilding_on_demand", extra={"context_key": context_key})
                
                # Extract metadata from the restored state to build the new meta
                parent_agent_id = state_data.get("parent_agent_id")
                # agent_id should be inferred from the key or retrieved from the state
                agent_id_from_state = state_data.get("agent_id", context_key.replace("_context_ref", "").replace("_", "").capitalize())

                rebuilt_sub_context: SubContext = {
                    "meta": {
                        "run_id": target_context["meta"]["run_id"],
                        "agent_id": agent_id_from_state,
                        "parent_agent_id": parent_agent_id,
                    },
                    "state": copy.deepcopy(state_data), # Deep copy the restored state
                    "runtime_objects": {}, # Runtime objects are always new and empty
                    "refs": { # References must point to the current new RunContext and TeamState
                        "run": target_context,
                        "team": target_context["team_state"],
                    }
                }
                
                # Place the rebuilt complete SubContext object back into run_context
                target_context["sub_context_refs"][context_key] = rebuilt_sub_context
                logger.info("sub_context_rebuilt", extra={"context_key": context_key})

            else:
                 logger.warning("sub_context_data_not_dict_cannot_rebuild", extra={"context_key": context_key})

    # 3. Subsequent cleanup logic (unchanged)
    # --- Post-injection Cleanup: Finalize any interrupted states from previous session ---
    team_state = target_context.get("team_state")
    if team_state:
        # Finalize any 'running' turns and their sub-components
        if team_state.get("turns"):
            for turn in team_state.get("turns", []):
                # Finalize turn if it was running
                if turn.get("status") == "running":
                    turn["status"] = "interrupted"
                    turn["error_details"] = "This action was active when the previous session ended and could not be completed."
                    turn["end_time"] = datetime.now(timezone.utc).isoformat()
                
                # Deeper check for running LLM interactions within the turn
                llm_interaction = turn.get("llm_interaction")
                if llm_interaction and llm_interaction.get("status") == "running":
                    llm_interaction["status"] = "error"
                    llm_interaction.setdefault("error", {})["message"] = "LLM interaction was interrupted by session termination."
                    
                    for attempt in llm_interaction.get("attempts", []):
                        if attempt.get("status") in ["pending", "running"]:
                            attempt["status"] = "failed"
                            attempt["error"] = "Run was interrupted during LLM stream."
        
        # Reset principal flow flag
        if team_state.get("is_principal_flow_running") is True:
            team_state["is_principal_flow_running"] = False
            logger.warning("principal_flow_running_reset_on_restore")

    logger.info("post_restoration_cleanup_complete", extra={"run_id": target_context['meta']['run_id']})
