import logging
import json
from typing import Dict, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import Event, EventQueue
from a2a.types import SendMessageRequest, Message

logger = logging.getLogger(__name__)

class InMemoryA2ARouter:
    _instance = None
    _executors: Dict[str, AgentExecutor] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InMemoryA2ARouter, cls).__new__(cls)
        return cls._instance

    def register_agent(self, agent_name: str, executor: AgentExecutor):
        self._executors[agent_name] = executor
        logger.info(f"In-memory A2A agent '{agent_name}' registered.")

    def is_internal(self, agent_name: str) -> bool:
        return agent_name in self._executors

    async def route_intent(self, agent_name: str, intent_request: SendMessageRequest) -> Dict[str, Any]:
        """模拟 A2A 请求-响应流程，完全在内存中进行"""
        if agent_name not in self._executors:
            return {"status": "failure", "error": f"Agent '{agent_name}' not found."}

        executor = self._executors[agent_name]
        
        # 1. 模拟 A2A Server 的 RequestContext 和 EventQueue
        context = RequestContext(
            task_id=intent_request.id,
            raw_request=intent_request.model_dump(),
            message=intent_request.params.message,
            # 其他字段可以根据需要填充
        )
        event_queue = EventQueue()

        # 2. 直接调用 executor 的 execute 方法
        try:
            await executor.execute(context, event_queue)
        except Exception as e:
            logger.error(f"In-memory executor for {agent_name} failed", exc_info=True)
            return {"status": "failure", "error": str(e)}

        # 3. 从 EventQueue 中收集所有事件并组装成最终的 A2A Result
        events: list[Event] = []
        while not event_queue.is_empty():
            events.append(await event_queue.dequeue_event())
        
        # 简化版的结果组装：只关心 message 和 error 事件
        final_parts = []
        error_message = None
        for event in events:
            if event.event_type == 'message':
                final_parts.extend(event.message.parts)
            elif event.event_type == 'error':
                error_message = event.message
        
        if error_message:
            return {"status": "failure", "error": error_message}

        final_message = Message(role='agent', parts=final_parts)
        return {"status": "success", "message": final_message.model_dump()}

# 创建一个全局单例
A2A_ROUTER = InMemoryA2ARouter()