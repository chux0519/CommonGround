import json
import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message
from a2a.types import TaskStatusUpdateEvent, TaskStatus, TaskState

# 导入您自己的 RAG 工具节点
from agent_core.nodes.custom_nodes.list_rag_sources_tool import ListRAGSourcesNode
from agent_core.nodes.custom_nodes.rag_query_node import RAGQueryNode
from agent_core.framework.tool_registry import get_tool_by_name

logger = logging.getLogger(__name__)

class SmartRAG_A2A_Executor(AgentExecutor):
    """
    这个 Executor 接收 A2A 请求，并将其路由到 Common Ground 的 RAG 工具节点。
    """
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        try:
            # 1. 从 A2A 请求中解析出我们的自定义参数和 action_name
            metadata = context.metadata
            action_name = metadata.get('skill_id')
            tool_params = metadata.get('parameters', {})
            run_id = metadata.get('context', {}).get('run_id')
            project_id = metadata.get('context', {}).get('project_id')

            # 获取对应的工具节点类
            tool_info = get_tool_by_name(action_name)
            if not tool_info or not tool_info.get('node_class'):
                raise ValueError(f"Tool '{action_name}' not found or has no associated node class.")
            ToolNodeClass = tool_info['node_class']

        except (KeyError, ValueError) as e:
            await event_queue.enqueue_error_event(f"Invalid A2A intent for SmartRAG: {e}")
            await event_queue.enqueue_done_event()
            return

        # 2. 构造运行工具节点所需的最小化上下文 (mock_context)
        # 注意：RAG 工具需要 project_id，所以我们也要传递它
        mock_sub_context = {
            "meta": {"run_id": run_id, "agent_id": "A2A_SmartRAG"},
            "state": {"current_action": tool_params},
            "refs": {
                "run": {
                    "meta": {"run_id": run_id},
                    "project_id": project_id,
                    # 如果工具依赖 RAGFederationService，可能需要模拟更多上下文
                    # 但 RAGFederationService 是单例，应该能直接工作
                    "runtime": {}, 
                    "config": {}
                }
            }
        }

        # 3. 实例化并执行工具节点
        tool_node_instance = ToolNodeClass()
        tool_node_instance._tool_info = tool_info # 确保工具信息被设置

        prep_res = await tool_node_instance.prep_async(mock_sub_context)
        exec_res = await tool_node_instance.exec_async(prep_res)
        res_txt = json.dumps(exec_res)
        # 注意: BaseToolNode的post_async会将结果放入inbox, 我们在这里不需要, 
        # 所以我们直接使用 exec_res。如果工具逻辑复杂，可能需要调整。
        # 对于 RAGQueryNode，exec_res 就是我们想要的结果。

        # 4. 将结果包装成 A2A 事件
        await event_queue.enqueue_event(new_agent_text_message(res_txt))
        
        # 5. 发送 done 事件
        done_event = TaskStatusUpdateEvent(
            task_id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.completed),
            final=True
        )
        await event_queue.enqueue_event(done_event)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        cancel_event = TaskStatusUpdateEvent(
            task_id=context.task_id,
            context_id=context.context_id,
            status=TaskStatus(state=TaskState.canceled, message="Task was cancelled."),
            final=True
        )
        await event_queue.enqueue_event(cancel_event)