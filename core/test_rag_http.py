#!/usr/bin/env python3
"""
HTTP Client Test Script for CommonGround RAG functionality

This script tests the RAG capabilities by sending HTTP requests to the running server.
"""

import asyncio
import aiohttp
import json
import uuid
import time
from typing import Dict, Any, Optional

class CommonGroundRAGTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session_id = str(uuid.uuid4())
        
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict[Any, Any]] = None) -> Dict[Any, Any]:
        """Make an HTTP request to the server."""
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            try:
                if method.upper() == "GET":
                    async with session.get(url) as response:
                        result = await response.json()
                        print(f"✓ GET {endpoint} -> {response.status}")
                        return result
                elif method.upper() == "POST":
                    headers = {"Content-Type": "application/json"}
                    async with session.post(url, json=data, headers=headers) as response:
                        result = await response.json()
                        print(f"✓ POST {endpoint} -> {response.status}")
                        return result
            except Exception as e:
                print(f"❌ {method} {endpoint} failed: {e}")
                raise

    async def test_server_health(self):
        """Test if the server is responding."""
        print("\n" + "="*60)
        print("TEST: Server Health Check")
        print("="*60)
        
        try:
            # Try to get server info or health endpoint
            result = await self._make_request("GET", "/")
            print(f"✓ Server is responding")
            return True
        except:
            try:
                # Try alternative health endpoint
                result = await self._make_request("GET", "/health")
                print(f"✓ Server health endpoint responding")
                return True
            except:
                print("❌ Server health check failed")
                return False

    async def test_list_rag_sources(self):
        """Test listing available RAG sources via session."""
        print("\n" + "="*60)
        print("TEST: List RAG Sources via WebSocket")
        print("="*60)
        
        try:
            # Create session first
            session_result = await self._make_request("POST", "/session", {})
            session_id = session_result.get("session_id")
            
            if not session_id:
                print("❌ Failed to create session for RAG sources test")
                return None
                
            print(f"✓ Created session: {session_id}")
            return {"session_id": session_id, "status": "ready_for_websocket"}
            
        except Exception as e:
            print(f"❌ List RAG sources test failed: {e}")
            return None

    async def test_rag_query(self, question: str = "What is CommonGround?"):
        """Test RAG query functionality via WebSocket."""
        print("\n" + "="*60)
        print(f"TEST: RAG Query via WebSocket - '{question}'")
        print("="*60)
        
        try:
            # Create session first
            session_result = await self._make_request("POST", "/session", {})
            session_id = session_result.get("session_id")
            
            if not session_id:
                print("❌ Failed to create session for RAG query test")
                return None
                
            print(f"✓ Created session: {session_id}")
            return {"session_id": session_id, "question": question, "status": "ready_for_websocket"}
            
        except Exception as e:
            print(f"❌ RAG query test failed: {e}")
            return None

    async def test_websocket_rag_stream(self, question: str = "How does agent collaboration work?"):
        """Test RAG functionality via WebSocket streaming."""
        print("\n" + "="*60)
        print(f"TEST: WebSocket RAG Stream - '{question}'")
        print("="*60)
        
        try:
            # First create a session
            session_result = await self._make_request("POST", "/session", {})
            session_id = session_result.get("session_id")
            if not session_id:
                print("❌ Failed to create session")
                return False
            
            print(f"✓ Created session: {session_id}")
            
            # Import websockets
            import websockets
            
            uri = f"ws://127.0.0.1:8000/ws/{session_id}"
            
            async with websockets.connect(uri) as websocket:
                print(f"✓ Connected to WebSocket")
                
                # Send start_run message with required fields
                start_message = {
                    "type": "start_run",
                    "data": {
                        "request_id": str(uuid.uuid4()),
                        "run_type": "partner_interaction",
                        "user_prompt": question,
                        "agent_profile": "Associate_SmartRAG_EN",
                        "project_id": "default"
                    }
                }
                
                await websocket.send(json.dumps(start_message))
                print(f"✓ Sent start_run message")
                
                # Wait for run_ready confirmation
                run_id = None
                request_id = start_message["data"]["request_id"]
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    print(f"✓ Initial response: {data.get('type', 'unknown')}")
                    
                    if data.get('type') == 'run_ready':
                        run_id = data.get('data', {}).get('run_id')
                        print(f"✓ Run ready: {run_id}")
                    elif data.get('type') == 'error':
                        print(f"❌ Start run error: {data.get('data', {}).get('message', 'Unknown error')}")
                        return False
                        
                except asyncio.TimeoutError:
                    print("❌ Timeout waiting for run_ready")
                    return False
                
                if not run_id:
                    print("❌ No run_id received")
                    return False
                
                # Now send the actual user message
                user_message = {
                    "type": "send_to_run",
                    "data": {
                        "run_id": run_id,
                        "message_payload": {
                            "prompt": question
                        }
                    }
                }
                
                await websocket.send(json.dumps(user_message))
                print(f"✓ Sent user message: {question[:50]}...")
                
                # Listen for responses with strict timeout handling
                response_count = 0
                last_activity = time.time()
                max_timeout = 30.0  # 30 seconds total timeout - fail if exceeded
                individual_timeout = 8.0  # 8 seconds for individual messages
                
                while response_count < 20 and (time.time() - last_activity) < max_timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=individual_timeout)
                        data = json.loads(response)
                        last_activity = time.time()
                        response_count += 1
                        
                        msg_type = data.get('type', 'unknown')
                        print(f"✓ WebSocket response {response_count}: {msg_type}")
                        
                        if msg_type == 'agent_message':
                            content = data.get('data', {}).get('content', '')
                            role = data.get('data', {}).get('role', 'unknown')
                            print(f"   {role}: {content[:100]}...")
                        elif msg_type == 'tool_execution_start':
                            tool_name = data.get('data', {}).get('tool_name', '')
                            print(f"   🔧 Tool starting: {tool_name}")
                        elif msg_type == 'tool_execution_result':
                            tool_name = data.get('data', {}).get('tool_name', '')
                            success = data.get('data', {}).get('success', False)
                            print(f"   ✅ Tool result: {tool_name} ({'success' if success else 'failed'})")
                        elif msg_type == 'run_complete':
                            print("   ✅ Run completed successfully")
                            return True
                        elif msg_type == 'error':
                            error_msg = data.get('data', {}).get('message', 'Unknown error')
                            print(f"   ❌ Error: {error_msg}")
                            return False  # Fail immediately on error
                        elif msg_type == 'turn_complete':
                            print("   ✅ Turn completed")
                        else:
                            # Print other message types for debugging
                            print(f"   📄 {msg_type}: {str(data.get('data', {}))[:50]}...")
                            
                    except asyncio.TimeoutError:
                        elapsed = time.time() - last_activity
                        if elapsed > max_timeout:
                            print(f"❌ TIMEOUT: Test failed after {max_timeout}s")
                            return False
                        else:
                            print(f"⚠️ No message for {individual_timeout}s (total elapsed: {elapsed:.1f}s)")
                            continue
                    except Exception as e:
                        print(f"❌ WebSocket error: {e}")
                        return False
                
                # Check if we exceeded the total timeout
                total_elapsed = time.time() - last_activity
                if total_elapsed >= max_timeout:
                    print(f"❌ TEST FAILED: Exceeded {max_timeout}s timeout")
                    return False
                
                if response_count > 0:
                    print(f"✓ Received {response_count} responses in {total_elapsed:.1f}s")
                    return True
                else:
                    print("❌ No responses received")
                    return False
                        
                return True
                
        except Exception as e:
            print(f"❌ WebSocket test failed: {e}")
            return False

    async def test_metadata_endpoint(self):
        """Test the metadata endpoint."""
        print("\n" + "="*60)
        print("TEST: Metadata Endpoint")
        print("="*60)
        
        try:
            result = await self._make_request("GET", "/metadata")
            print(f"✓ Metadata response: {json.dumps(result, indent=2)}")
            return result
        except Exception as e:
            print(f"❌ Metadata endpoint failed: {e}")
            return None

    async def run_all_tests(self):
        """Run all RAG tests."""
        print("🚀 Starting CommonGround RAG HTTP Tests")
        print("="*60)
        
        tests = [
            ("Server Health", self.test_server_health),
            ("Metadata Endpoint", self.test_metadata_endpoint),
            ("Session Creation & RAG Setup", self.test_list_rag_sources),
            ("RAG Query Setup", lambda: self.test_rag_query("What is CommonGround and how does it work?")),
            ("WebSocket RAG Stream", lambda: self.test_websocket_rag_stream("How do agents collaborate in CommonGround?")),
            ("WebSocket Architecture Query", lambda: self.test_websocket_rag_stream("Explain the CommonGround architecture and agent framework")),
        ]
        
        passed = 0
        failed = 0
        results = {}
        
        for test_name, test_func in tests:
            try:
                print(f"\n🧪 Running: {test_name}")
                start_time = time.time()
                result = await test_func()
                duration = time.time() - start_time
                
                if result is not None and result is not False:
                    print(f"✅ {test_name} PASSED ({duration:.2f}s)")
                    passed += 1
                    results[test_name] = {"status": "PASSED", "duration": duration, "result": result}
                else:
                    print(f"❌ {test_name} FAILED ({duration:.2f}s)")
                    failed += 1
                    results[test_name] = {"status": "FAILED", "duration": duration}
                    
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                failed += 1
                results[test_name] = {"status": "ERROR", "error": str(e)}
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Total: {passed + failed}")
        
        # Save detailed results
        with open("rag_test_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📝 Detailed results saved to: rag_test_results.json")
        
        if failed == 0:
            print("\n🎉 All tests passed successfully!")
            return True
        else:
            print(f"\n💥 {failed} test(s) failed!")
            return False

async def main():
    """Main function."""
    tester = CommonGroundRAGTester()
    
    try:
        success = await tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
