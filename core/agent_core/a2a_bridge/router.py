import logging
import json
import asyncio
from typing import Dict, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import Event, EventQueue
from a2a.server.events.event_consumer import QueueClosed
from a2a.types import TaskStatusUpdateEvent, Message, SendMessageRequest

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
        """[REVISED] 模拟 A2A 请求-响应流程，并正确地消费事件流。"""
        if agent_name not in self._executors:
            return {"status": "failure", "error": f"Agent '{agent_name}' not found."}

        executor = self._executors[agent_name]
        
        # 1. 模拟 RequestContext 和 EventQueue (保持不变)
        context = RequestContext(
            task_id=intent_request.id,
            raw_request=intent_request.model_dump(by_alias=True),
            message=intent_request.params.message,
        )
        event_queue = EventQueue()

        # 2. 创建一个后台任务来运行 Executor
        #    这让 executor 可以和我们的消费循环并行运行
        agent_task = asyncio.create_task(executor.execute(context, event_queue))

        # --- START FIX: 正确的事件消费循环 ---
        
        # 3. 从 EventQueue 中收集所有事件，直到收到 'final' 事件
        events: list[Event] = []
        try:
            while True:
                try:
                    # 使用带超时的 get 来防止永久阻塞，并允许检查 agent_task 的状态
                    event = await asyncio.wait_for(event_queue.dequeue_event(), timeout=1.0)
                    events.append(event)
                    
                    # 检查是否是结束事件
                    is_final = False
                    if isinstance(event, TaskStatusUpdateEvent):
                        is_final = event.final
                    elif isinstance(event, Message):
                        # 在非流式场景下，单个 Message 也是结束信号
                        is_final = True

                    if is_final:
                        break # 收到结束信号，退出循环

                except asyncio.TimeoutError:
                    # 超时后检查 agent_task 是否已经意外结束（例如，出错了）
                    if agent_task.done():
                        # 如果任务已结束但我们没收到 final event，说明有异常
                        # agent_task.exception() 会重新抛出异常
                        if agent_task.exception():
                            raise agent_task.exception()
                        break # 任务正常结束，退出循环
                    continue # 任务还在运行，继续等待下一个事件
                
        except QueueClosed:
            # 如果 executor 调用了 queue.close()，这也是一个合法的退出条件
            logger.info(f"Event queue for agent '{agent_name}' was closed.")
        except Exception as e:
            logger.error(f"Error while consuming events for '{agent_name}'", exc_info=True)
            return {"status": "failure", "error": f"Executor task failed: {e}"}
        finally:
            # 确保 agent_task 被清理
            if not agent_task.done():
                agent_task.cancel()
            await event_queue.close() # 确保队列被关闭
        
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