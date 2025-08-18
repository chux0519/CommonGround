# agent_core/a2a_bridge/router.py

import logging
import json
import asyncio
from typing import Dict, Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import Event, EventQueue, EventConsumer
from a2a.server.tasks import TaskManager, ResultAggregator, InMemoryTaskStore
from a2a.types import Message, SendMessageRequest, TextPart

logger = logging.getLogger(__name__)

class InMemoryA2ARouter:
    _instance = None
    _executors: Dict[str, AgentExecutor] = {}
    _task_store = InMemoryTaskStore() # A store for the dummy task

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
        """[REVISED] 使用 A2A SDK 的标准组件来模拟请求-响应流程。"""
        if agent_name not in self._executors:
            return {"status": "failure", "error": f"Agent '{agent_name}' not found."}

        executor = self._executors[agent_name]
        
        # 1. 模拟 RequestContext 和 EventQueue (保持不变)
        context = RequestContext(
            request=intent_request.params,
            task_id=intent_request.id
        )
        event_queue = EventQueue()

        # 2. 创建一个后台任务来运行 Executor (Producer)
        agent_task = asyncio.create_task(executor.execute(context, event_queue))

        # 3. 使用 SDK 的标准组件来消费事件 (Consumer)
        try:
            # 创建一个临时的 TaskManager 来聚合状态
            task_manager = TaskManager(
                task_id=context.task_id,
                context_id=context.context_id,
                task_store=self._task_store,
                initial_message=context.message,
            )
            
            # ResultAggregator 封装了标准的事件消费逻辑
            result_aggregator = ResultAggregator(task_manager)
            
            # EventConsumer 是从队列中安全读取事件的标准方式
            consumer = EventConsumer(queue=event_queue)
            
            # 这是一个关键步骤：将 producer task 的完成状态（特别是异常）与 consumer 关联起来
            agent_task.add_done_callback(consumer.agent_task_callback)
            
            # 使用 consume_all 等待并处理所有事件，直到流结束
            final_result = await result_aggregator.consume_all(consumer)

            # 4. 从最终结果中提取消息
            if isinstance(final_result, Message):
                return {"status": "success", "message": final_result.model_dump()}
            elif final_result: # It might be a Task object
                # 尝试从 Task 历史中找到最后一条 agent 消息
                last_agent_message = next((msg for msg in reversed(final_result.history or []) if msg.role == 'agent'), None)
                if last_agent_message:
                    return {"status": "success", "message": last_agent_message.model_dump()}

            # 如果没有直接的消息，从 executor 的结果中构建一个
            # (这个 fallback 可能不需要，但为了健壮性保留)
            if hasattr(result_aggregator, '_message') and result_aggregator._message:
                 return {"status": "success", "message": result_aggregator._message.model_dump()}
            
            return {"status": "failure", "error": "Executor finished but did not produce a final message."}

        except Exception as e:
            logger.error(f"Error while routing intent for '{agent_name}'", exc_info=True)
            return {"status": "failure", "error": f"Executor task failed: {e}"}
        finally:
            # 确保后台任务和队列被清理
            if not agent_task.done():
                agent_task.cancel()
            if not event_queue.is_closed():
                await event_queue.close()


# 创建一个全局单例
A2A_ROUTER = InMemoryA2ARouter()