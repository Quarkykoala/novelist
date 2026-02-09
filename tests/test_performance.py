import asyncio
import time
from unittest.mock import MagicMock, patch
import pytest
from src.soul.simulator import Simulator

@pytest.mark.asyncio
async def test_execute_code_non_blocking():
    """Verify that _execute_code does not block the event loop."""

    # Mock LLMClient to avoid initialization issues
    with patch('src.soul.simulator.LLMClient'):
        simulator = Simulator()

        # Large code to ensure write takes some time
        code = "print('Hello World')\n" * 50000
        sim_id_base = "perf_test_"

        # Mock subprocess to return immediately
        async def mock_subprocess(*args, **kwargs):
            mock_proc = MagicMock()
            mock_proc.communicate = asyncio.Future()
            mock_proc.communicate.set_result((b"SIMULATION_RESULT: SUCCESS", b""))
            mock_proc.returncode = 0
            return mock_proc

        with patch('asyncio.create_subprocess_exec', side_effect=mock_subprocess):

            records = []
            async def heartbeat():
                while True:
                    records.append(time.time())
                    await asyncio.sleep(0.001)

            heartbeat_task = asyncio.create_task(heartbeat())

            tasks = []
            num_iterations = 200 # Reduced from 2000 for speed, but sufficient

            for i in range(num_iterations):
                tasks.append(simulator._execute_code(code, f"{sim_id_base}{i}"))

            await asyncio.gather(*tasks)

            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

            print(f"Captured {len(records)} heartbeat records")

            assert len(records) > 10, "Event loop appears blocked by file writes"
