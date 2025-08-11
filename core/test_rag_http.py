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


    async def test_websocket_rag_stream(self, question: str = "How does agent collaboration work?"):
        """
        [REVISED] Tests the full Principal -> Associate (A2A) -> Principal RAG flow.
        """
        print("\n" + "="*60)
        print(f"TEST: Full A2A RAG Flow - '{question}'")
        print("="*60)
        
        try:
            # Step 1 & 2: Create session and connect to WebSocket (no changes needed)
            session_result = await self._make_request("POST", "/session", {})
            session_id = session_result.get("session_id")
            if not session_id:
                print("❌ Failed to create session")
                return False
            print(f"✓ Created session: {session_id}")
            
            import websockets
            uri = f"ws://127.0.0.1:8000/ws/{session_id}"
            
            async with websockets.connect(uri) as websocket:
                print(f"✓ Connected to WebSocket")
                
                # Step 3: Start a 'principal_direct' run (no changes needed)
                start_message = {
                    "type": "start_run",
                    "data": {
                        "request_id": str(uuid.uuid4()),
                        "run_type": "principal_direct",
                        "project_id": "default",
                        "cli_mode_default_profiles": ["Associate_SmartRAG_EN"]
                    }
                }
                await websocket.send(json.dumps(start_message))
                print(f"✓ Sent start_run message for a DIRECT PRINCIPAL run")
                
                # Step 4: Wait for run_ready (no changes needed)
                run_id = None
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    if data.get('type') == 'run_ready':
                        run_id = data.get('data', {}).get('run_id')
                        print(f"✓ Run ready: {run_id}")
                    else:
                        print(f"❌ Start run error: {data}")
                        return False
                except asyncio.TimeoutError:
                    print("❌ Timeout waiting for run_ready")
                    return False
                if not run_id: return False
                
                # Step 5: Send initial directive to Principal (no changes needed)
                user_message = {
                    "type": "send_to_run",
                    "data": {
                        "run_id": run_id,
                        "message_payload": {"prompt": question}
                    }
                }
                await websocket.send(json.dumps(user_message))
                print(f"✓ Sent initial directive to Principal: {question[:50]}...")
                
                # +++ START: REVISED LISTENING LOGIC +++
                
                print("\n--- Listening for full A2A dispatch and result cycle ---")
                
                # We now have two success criteria to meet in order
                plan_created = False
                dispatch_completed = False
                
                max_timeout = 90.0  # Increased total timeout to 90 seconds for the full flow
                start_time = time.time()
                
                while not dispatch_completed and (time.time() - start_time) < max_timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=20.0) # Increased per-message timeout
                        data = json.loads(response)
                        
                        msg_type = data.get('type', 'unknown')
                        agent_id = data.get('agent_id', 'System')
                        
                        print(f"✓ WS Recv: {msg_type} from {agent_id}")

                        # Checkpoint 1: Principal creates the plan
                        if msg_type == 'llm_response' and agent_id == 'Principal':
                            tool_calls = data.get('data', {}).get('tool_calls', [])
                            if any(tc.get('function', {}).get('name') == 'manage_work_modules' for tc in tool_calls):
                                print("  -> CHECKPOINT 1: Principal called 'manage_work_modules'. Plan created.")
                                plan_created = True

                        # Checkpoint 2: Principal receives the RAG result
                        if msg_type == 'llm_response' and agent_id == 'Principal' and plan_created:
                             # The final success is seeing the Principal's *reaction* to the RAG result
                             content = data.get('data', {}).get('content', '')
                             if "search_results" in content or "relevant documents" in content:
                                 print("  -> CHECKPOINT 2: Principal received and is processing the RAG results from the Associate.")
                                 dispatch_completed = True

                        if msg_type == 'error':
                            print(f"❌ Error received: {data.get('data', {}).get('message')}")
                            return False # Exit on any error

                    except asyncio.TimeoutError:
                        print(f"⏳ Waiting... ({(time.time() - start_time):.1f}s elapsed)")
                        continue
                    except Exception as e:
                        print(f"❌ WebSocket error: {e}")
                        return False
                
                # Final check after the loop
                if dispatch_completed:
                    print("\n🎉 Test PASSED: Full Principal -> A2A Associate -> Principal communication cycle was successful!")
                    return True
                else:
                    print(f"\n❌ TEST FAILED: Did not complete the full A2A dispatch and result cycle within {max_timeout}s.")
                    return False
                # +++ END: REVISED LISTENING LOGIC +++
                
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
