import threading
import time
import unittest

from sam3_demo.session_runtime import (
    SessionError,
    SessionExpired,
    SessionRegistry,
    resume_id_for_identity,
    resolve_client_ip,
    validate_trusted_proxy_cidrs,
)


class FakeClock:
    def __init__(self, value=0.0):
        self.value = float(value)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class ClientIpResolutionTests(unittest.TestCase):
    def test_untrusted_peer_ignores_forwarded_and_real_ip(self):
        self.assertEqual(
            resolve_client_ip(
                "198.51.100.20",
                {
                    "X-Forwarded-For": "203.0.113.9",
                    "X-Real-IP": "203.0.113.10",
                },
                ["10.0.0.0/8"],
            ),
            "198.51.100.20",
        )

    def test_single_and_multi_level_trusted_proxy_chain(self):
        trusted = ["10.0.0.0/8", "192.0.2.0/24"]
        self.assertEqual(
            resolve_client_ip("10.1.1.1", {"X-Forwarded-For": "198.51.100.4"}, trusted),
            "198.51.100.4",
        )
        self.assertEqual(
            resolve_client_ip(
                "10.1.1.1",
                {"X-Forwarded-For": "198.51.100.4, 192.0.2.8, 10.2.2.2"},
                trusted,
            ),
            "198.51.100.4",
        )

    def test_malformed_forwarded_chain_falls_back_to_peer(self):
        trusted = ["10.0.0.0/8"]
        for value in ("203.0.113.4, nope", "203.0.113.4,", ",203.0.113.4", ""):
            self.assertEqual(
                resolve_client_ip("10.0.0.8", {"X-Forwarded-For": value}, trusted),
                "10.0.0.8",
            )

    def test_ipv6_is_canonicalized_and_proxy_chain_is_supported(self):
        self.assertEqual(resolve_client_ip("2001:0DB8:0:0::1"), "2001:db8::1")
        self.assertEqual(
            resolve_client_ip(
                "2001:db8:ffff::1",
                {"x-forwarded-for": "2001:db8::44, 2001:db8:1::2"},
                ["2001:db8:ffff::/48", "2001:db8:1::/48"],
            ),
            "2001:db8::44",
        )

    def test_empty_proxy_configuration_means_direct_connection(self):
        self.assertEqual(resolve_client_ip("198.51.100.20", trusted_proxy_cidrs=""), "198.51.100.20")

    def test_invalid_peer_or_proxy_configuration_is_explicit(self):
        with self.assertRaises(ValueError):
            resolve_client_ip("not-an-ip")
        with self.assertRaises(ValueError):
            resolve_client_ip("10.0.0.1", trusted_proxy_cidrs=["not-a-cidr"])
        with self.assertRaises(ValueError):
            validate_trusted_proxy_cidrs("10.0.0.0/8,not-a-cidr")
        validate_trusted_proxy_cidrs("")


class SessionRegistryTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.cleaned = []
        self.registry = SessionRegistry(
            clock=self.clock,
            idle_seconds=10,
            max_sessions=10,
            secret=b"test-session-secret",
            cleanup_callback=self.cleaned.append,
        )

    def tearDown(self):
        self.registry.shutdown()

    def test_registry_rejects_nonfinite_timing_values(self):
        for field in ("idle_seconds", "sweep_interval"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, value=value):
                    kwargs = {field: value}
                    with self.assertRaises(ValueError):
                        SessionRegistry(**kwargs)

    def test_state_is_opaque_and_identity_tuple_isolated(self):
        first = self.registry.bind("browser-a", "203.0.113.1")
        same = self.registry.bind("browser-a", "203.0.113.1")
        other_ip = self.registry.bind("browser-a", "203.0.113.2")
        other_hash = self.registry.bind("browser-b", "203.0.113.1")
        self.assertEqual(first["session_id"], same["session_id"])
        self.assertNotEqual(first["session_id"], other_ip["session_id"])
        self.assertNotEqual(first["session_id"], other_hash["session_id"])
        self.assertNotIn("browser-a", repr(first))
        self.assertNotIn("203.0.113.1", repr(first))
        self.assertNotEqual(first["session_id"], first["owner_token"])
        self.assertEqual(self.registry.snapshot()["count"], 3)

    def test_validate_rejects_wrong_ip_hash_and_token(self):
        state = self.registry.bind("browser-a", "203.0.113.1")
        self.assertEqual(
            self.registry.validate(state, "browser-a", "203.0.113.1").session_id,
            state["session_id"],
        )
        for supplied_hash, supplied_ip in (
            ("browser-b", "203.0.113.1"),
            ("browser-a", "203.0.113.2"),
        ):
            with self.assertRaises(SessionError):
                self.registry.validate(state, supplied_hash, supplied_ip)
        tampered = dict(state)
        tampered["owner_token"] = "0" * len(state["owner_token"])
        with self.assertRaises(SessionError):
            self.registry.validate(tampered, "browser-a", "203.0.113.1")

    def test_validate_owner_and_owner_lease_require_bound_identity(self):
        a = self.registry.bind("browser-a", "203.0.113.1")
        b = self.registry.bind("browser-b", "203.0.113.1")
        self.assertEqual(
            self.registry.validate_owner(a["session_id"], "browser-a", "203.0.113.1").session_id,
            a["session_id"],
        )
        with self.assertRaises(SessionError):
            self.registry.validate_owner(a["session_id"], "browser-b", "203.0.113.1")
        with self.assertRaises(SessionError):
            self.registry.validate_owner(b["session_id"], "browser-a", "203.0.113.1")
        with self.registry.owner_lease(a["session_id"], "browser-a", "203.0.113.1") as record:
            self.assertEqual(record.in_flight, 1)

    def test_reentrant_lease_and_active_record_is_not_reaped(self):
        state = self.registry.bind("browser-a", "203.0.113.1")
        with self.registry.lease(state, "browser-a", "203.0.113.1") as record:
            with self.registry.lease(state, "browser-a", "203.0.113.1") as nested:
                self.assertIs(record, nested)
                self.assertEqual(record.in_flight, 1)
            self.clock.advance(100)
            self.assertEqual(self.registry.reap_expired(), 0)
            self.assertEqual(self.registry.snapshot()["count"], 1)
        self.clock.advance(10)
        self.assertEqual(self.registry.reap_expired(), 1)
        self.assertEqual(len(self.cleaned), 1)
        with self.assertRaises(SessionExpired):
            self.registry.validate(state, "browser-a", "203.0.113.1")

    def test_same_session_leases_serialize_across_threads(self):
        state = self.registry.bind("browser-a", "203.0.113.1")
        first_entered = threading.Event()
        release_first = threading.Event()
        second_attempting = threading.Event()
        second_entered = threading.Event()

        def first():
            with self.registry.lease(state, "browser-a", "203.0.113.1"):
                first_entered.set()
                release_first.wait(2)

        def second():
            first_entered.wait(2)
            second_attempting.set()
            with self.registry.lease(state, "browser-a", "203.0.113.1"):
                second_entered.set()

        first_thread = threading.Thread(target=first)
        second_thread = threading.Thread(target=second)
        first_thread.start()
        second_thread.start()
        self.assertTrue(first_entered.wait(1))
        self.assertTrue(second_attempting.wait(1))
        self.assertFalse(second_entered.wait(0.1))
        release_first.set()
        first_thread.join(2)
        second_thread.join(2)
        self.assertFalse(first_thread.is_alive())
        self.assertFalse(second_thread.is_alive())
        self.assertTrue(second_entered.is_set())

    def test_different_session_leases_run_in_parallel(self):
        first = self.registry.bind("browser-a", "203.0.113.1")
        second = self.registry.bind("browser-b", "203.0.113.1")
        first_entered = threading.Event()
        second_entered = threading.Event()
        release = threading.Event()

        def worker(state, browser_hash, entered):
            with self.registry.lease(state, browser_hash, "203.0.113.1"):
                entered.set()
                release.wait(2)

        threads = [
            threading.Thread(target=worker, args=(first, "browser-a", first_entered)),
            threading.Thread(target=worker, args=(second, "browser-b", second_entered)),
        ]
        for thread in threads:
            thread.start()
        self.assertTrue(first_entered.wait(1))
        self.assertTrue(second_entered.wait(1))
        release.set()
        for thread in threads:
            thread.join(2)
            self.assertFalse(thread.is_alive())

    def test_ttl_boundary_and_validation_touch(self):
        state = self.registry.bind("browser-a", "203.0.113.1")
        self.clock.advance(9.9)
        self.registry.validate(state, "browser-a", "203.0.113.1")
        self.clock.advance(9.9)
        self.assertEqual(self.registry.reap_expired(), 0)
        self.clock.advance(0.1)
        self.assertEqual(self.registry.reap_expired(), 1)

    def test_close_state_is_strict_and_cleanup_is_idempotent(self):
        state = self.registry.bind("browser-a", "203.0.113.1")
        tampered = dict(state)
        tampered["generation"] = 2
        self.assertFalse(self.registry.close_state(tampered))
        self.assertEqual(self.registry.snapshot()["count"], 1)
        self.assertTrue(self.registry.close_state(state))
        self.assertFalse(self.registry.close_state(state))
        self.assertEqual(len(self.cleaned), 1)
    def test_close_with_state_requires_exact_hash_and_ip(self):
        state = self.registry.bind("browser-a", "203.0.113.1")
        self.assertFalse(self.registry.close(state, "browser-b", "203.0.113.1"))
        self.assertFalse(self.registry.close(state, "browser-a", "203.0.113.2"))
        self.assertTrue(self.registry.close(state, "browser-a", "203.0.113.1"))
        self.assertFalse(self.registry.close(state, "browser-a", "203.0.113.1"))
        with self.assertRaises(SessionExpired):
            self.registry.validate(state, "browser-a", "203.0.113.1")



    def test_stale_delete_callback_cannot_close_rebound_session(self):
        old_state = self.registry.bind("browser-a", "203.0.113.1")
        self.assertTrue(self.registry.close_state(old_state))
        rebound = self.registry.bind("browser-a", "203.0.113.1")
        self.assertNotEqual(old_state["session_id"], rebound["session_id"])
        self.assertFalse(self.registry.close_state(old_state))
        self.assertEqual(
            self.registry.validate(
                rebound,
                "browser-a",
                "203.0.113.1",
            ).session_id,
            rebound["session_id"],
        )

    def test_close_active_defers_cleanup_until_lease_release(self):
        state = self.registry.bind("browser-a", "203.0.113.1")
        with self.registry.lease(state, "browser-a", "203.0.113.1"):
            self.assertTrue(self.registry.close_state(state))
            self.assertEqual(self.registry.snapshot()["count"], 1)
            self.assertEqual(self.cleaned, [])
            with self.assertRaises(SessionError):
                self.registry.validate(state, "browser-a", "203.0.113.1")
        self.assertEqual(self.registry.snapshot()["count"], 0)
        self.assertEqual(len(self.cleaned), 1)

    def test_capacity_evicts_oldest_inactive_only(self):
        registry = SessionRegistry(clock=self.clock, max_sessions=2, secret=b"capacity")
        try:
            first = registry.bind("a", "203.0.113.1")
            self.clock.advance(1)
            second = registry.bind("b", "203.0.113.1")
            self.clock.advance(1)
            third = registry.bind("c", "203.0.113.1")
            self.assertEqual(registry.snapshot()["count"], 2)
            with self.assertRaises(SessionExpired):
                registry.validate(first, "a", "203.0.113.1")
            self.assertEqual(registry.validate(second, "b", "203.0.113.1").session_id, second["session_id"])
            self.assertEqual(registry.validate(third, "c", "203.0.113.1").session_id, third["session_id"])
        finally:
            registry.shutdown()

    def test_bind_runs_expired_cleanup_before_capacity_error(self):
        cleaned = []
        registry = SessionRegistry(
            clock=self.clock,
            idle_seconds=10,
            max_sessions=2,
            secret=b"capacity-cleanup",
            cleanup_callback=cleaned.append,
        )
        try:
            expired = registry.bind("a", "203.0.113.1")
            active = registry.bind("b", "203.0.113.1")
            with registry.lease(active, "b", "203.0.113.1"):
                self.clock.advance(10)
                registry.max_sessions = 1
                with self.assertRaisesRegex(SessionError, "capacity"):
                    registry.bind("c", "203.0.113.1")
                self.assertEqual(
                    [record.session_id for record in cleaned],
                    [expired["session_id"]],
                )
        finally:
            registry.shutdown()

    def test_capacity_never_evicts_an_active_session(self):
        registry = SessionRegistry(clock=self.clock, max_sessions=1, secret=b"capacity-active")
        try:
            state = registry.bind("a", "203.0.113.1")
            with registry.lease(state, "a", "203.0.113.1"):
                with self.assertRaisesRegex(SessionError, "capacity"):
                    registry.bind("b", "203.0.113.1")
                self.assertEqual(registry.snapshot()["count"], 1)
        finally:
            registry.shutdown()


    def test_cleanup_callback_runs_outside_registry_lock(self):
        lock_states = []

        def callback(record):
            acquired = self.registry._lock.acquire(blocking=False)
            lock_states.append(acquired)
            if acquired:
                self.registry._lock.release()

        self.registry._cleanup_callback = callback
        state = self.registry.bind("browser-a", "203.0.113.1")
        self.assertTrue(self.registry.close_state(state))
        self.assertEqual(lock_states, [True])

    def test_cleanup_callback_failure_is_logged(self):
        registry = SessionRegistry(
            secret=b"cleanup-log",
            cleanup_callback=lambda record: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        state = registry.bind("browser-a", "203.0.113.1")
        try:
            with self.assertLogs("sam3_demo.session_runtime", level="ERROR") as captured:
                self.assertTrue(registry.close_state(state))
            self.assertIn("cleanup callback failed", " ".join(captured.output).lower())
        finally:
            registry.shutdown()

    def test_sweeper_expires_and_shutdown_is_idempotent(self):
        cleaned = threading.Event()
        registry = SessionRegistry(
            idle_seconds=0.02,
            sweep_interval=0.005,
            secret=b"sweeper",
            cleanup_callback=lambda record: cleaned.set(),
            start_sweeper=True,
        )
        state = registry.bind("browser-a", "203.0.113.1")
        try:
            self.assertTrue(cleaned.wait(1.0))
            self.assertEqual(registry.snapshot()["count"], 0)
            registry.shutdown()
            registry.shutdown()
            with self.assertRaises(SessionError):
                registry.bind("browser-a", "203.0.113.1")
            with self.assertRaises(SessionError):
                registry.validate(state, "browser-a", "203.0.113.1")
        finally:
            registry.shutdown()


class SessionRecoveryTests(unittest.TestCase):
    def test_resume_id_is_stable_across_registry_restarts_and_identity_scoped(self):
        first = SessionRegistry(secret=b"first")
        second = SessionRegistry(secret=b"second")
        try:
            a = first.bind("browser-a", "203.0.113.1")
            b = second.bind("browser-a", "203.0.113.1")
            self.assertEqual(a["resume_id"], b["resume_id"])
            self.assertEqual(
                a["resume_id"],
                resume_id_for_identity("browser-a", "203.0.113.1"),
            )
            self.assertNotEqual(
                a["resume_id"],
                resume_id_for_identity("browser-b", "203.0.113.1"),
            )
            self.assertNotEqual(
                a["resume_id"],
                resume_id_for_identity("browser-a", "203.0.113.2"),
            )
        finally:
            first.shutdown()
            second.shutdown()

    def test_ensure_recovers_restart_but_rejects_other_browser(self):
        old_registry = SessionRegistry(secret=b"old")
        stale = old_registry.bind("browser-a", "203.0.113.1")
        old_registry.shutdown()
        new_registry = SessionRegistry(secret=b"new")
        try:
            rebound, recovered = new_registry.ensure(
                stale,
                "browser-a",
                "203.0.113.1",
            )
            self.assertTrue(recovered)
            self.assertNotEqual(rebound["session_id"], stale["session_id"])
            with self.assertRaisesRegex(SessionError, "does not belong"):
                new_registry.ensure(stale, "browser-b", "203.0.113.1")
        finally:
            new_registry.shutdown()

    def test_concurrent_ensure_creates_one_rebound_session(self):
        registry = SessionRegistry(secret=b"concurrent")
        results = []
        barrier = threading.Barrier(20)

        def run():
            barrier.wait()
            results.append(registry.ensure(None, "browser-a", "203.0.113.1")[0])

        threads = [threading.Thread(target=run) for _ in range(20)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual({item["session_id"] for item in results}.__len__(), 1)
            self.assertEqual(registry.snapshot()["count"], 1)
        finally:
            registry.shutdown()


if __name__ == "__main__":
    unittest.main()
