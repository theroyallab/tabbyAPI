"""The disconnect wrapper owns and drains its two tasks when it exits."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from common.networking import run_with_request_disconnect


class RequestDisconnectLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.children = []
        self.finished = set()
        self.call_started = asyncio.Event()
        self.watch_started = asyncio.Event()
        self.call_release = asyncio.Event()
        self.watch_release = asyncio.Event()
        self.watch_task = None

    async def asyncTearDown(self):
        for task in self.children:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self.children, return_exceptions=True)

    async def call(self, error=None):
        self.call_started.set()
        try:
            await self.call_release.wait()
            if error:
                raise error
            return "result"
        finally:
            await asyncio.sleep(0)
            self.finished.add("call")

    async def watch(self, request):
        self.watch_task = asyncio.current_task()
        self.children.append(self.watch_task)
        self.watch_started.set()
        try:
            await self.watch_release.wait()
        finally:
            await asyncio.sleep(0)
            self.finished.add("watch")

    async def start_wrapper(self, error=None):
        task = asyncio.create_task(self.call(error))
        self.children.append(task)
        with patch("common.networking.request_disconnect_loop", self.watch):
            wrapper = asyncio.create_task(
                run_with_request_disconnect(SimpleNamespace(), task, "disconnected")
            )
            self.children.append(wrapper)
            await asyncio.wait_for(self.call_started.wait(), 1)
            await asyncio.wait_for(self.watch_started.wait(), 1)
        return task, wrapper

    async def test_outer_cancel_drains_both_children(self):
        task, wrapper = await self.start_wrapper()
        wrapper.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(wrapper, 1)
        self.assertTrue(task.done())
        self.assertTrue(self.watch_task.done())
        self.assertEqual(self.finished, {"call", "watch"})

    async def test_success_drains_disconnect_watcher(self):
        _, wrapper = await self.start_wrapper()
        self.call_release.set()
        self.assertEqual(await asyncio.wait_for(wrapper, 1), "result")
        self.assertTrue(self.watch_task.done())
        self.assertEqual(self.finished, {"call", "watch"})

    async def test_failure_drains_disconnect_watcher(self):
        _, wrapper = await self.start_wrapper(ValueError("call failed"))
        self.call_release.set()
        with self.assertRaisesRegex(ValueError, "call failed"):
            await asyncio.wait_for(wrapper, 1)
        self.assertTrue(self.watch_task.done())
        self.assertEqual(self.finished, {"call", "watch"})

    async def test_disconnect_drains_call_and_preserves_http_error(self):
        task, wrapper = await self.start_wrapper()
        self.watch_release.set()
        with self.assertRaises(HTTPException) as raised:
            await asyncio.wait_for(wrapper, 1)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail, "disconnected")
        self.assertTrue(task.done())
        self.assertEqual(self.finished, {"call", "watch"})

    async def test_cancelled_call_preserves_http_error_and_drains_watcher(self):
        task, wrapper = await self.start_wrapper()
        task.cancel()
        with self.assertRaises(HTTPException) as raised:
            await asyncio.wait_for(wrapper, 1)
        self.assertEqual(raised.exception.status_code, 422)
        self.assertTrue(self.watch_task.done())
        self.assertEqual(self.finished, {"call", "watch"})

    async def test_outer_cancel_waits_for_cooperative_child_cleanup(self):
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()

        async def delayed_cleanup():
            self.call_started.set()
            try:
                await self.call_release.wait()
            finally:
                cleanup_started.set()
                await cleanup_release.wait()
                self.finished.add("call")

        task = asyncio.create_task(delayed_cleanup())
        self.children.append(task)
        with patch("common.networking.request_disconnect_loop", self.watch):
            wrapper = asyncio.create_task(
                run_with_request_disconnect(SimpleNamespace(), task, "disconnected")
            )
            self.children.append(wrapper)
            try:
                await asyncio.wait_for(self.call_started.wait(), 1)
                await asyncio.wait_for(self.watch_started.wait(), 1)
                wrapper.cancel()
                await asyncio.wait_for(cleanup_started.wait(), 1)
                self.assertFalse(wrapper.done())
            finally:
                cleanup_release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(wrapper, 1)
        self.assertTrue(task.done())
        self.assertTrue(self.watch_task.done())
        self.assertEqual(self.finished, {"call", "watch"})
