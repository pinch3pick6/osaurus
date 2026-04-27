//
//  ModelLeaseTests.swift
//  osaurus
//

import Foundation
import Testing

@testable import OsaurusCore

@Suite(.serialized)
struct ModelLeaseTests {

    @Test func acquire_release_balances_to_zero() async {
        let lease = ModelLease.shared
        let name = "lease-test-\(UUID().uuidString)"

        await lease.acquire(name)
        await lease.acquire(name)
        var count = await lease.count(for: name)
        #expect(count == 2)

        await lease.release(name)
        await lease.release(name)
        count = await lease.count(for: name)
        #expect(count == 0)
    }

    @Test func waitForZero_resumes_when_count_drops() async {
        let lease = ModelLease.shared
        let name = "wait-test-\(UUID().uuidString)"
        await lease.acquire(name)

        let waiterFinished = AtomicBoolFlag()
        let waiterTask = Task {
            await lease.waitForZero(name)
            waiterFinished.set()
        }

        try? await Task.sleep(nanoseconds: 50_000_000)
        #expect(!waiterFinished.value)

        await lease.release(name)
        await waiterTask.value
        #expect(waiterFinished.value)
    }

    @Test func waitForZero_returns_immediately_when_no_lease() async {
        let lease = ModelLease.shared
        let name = "no-lease-\(UUID().uuidString)"
        await lease.waitForZero(name)
        // No assertion needed — reaching this line means no hang.
    }

    @Test func double_release_clamps_at_zero() async {
        let lease = ModelLease.shared
        let name = "clamp-test-\(UUID().uuidString)"
        await lease.acquire(name)
        await lease.release(name)
        await lease.release(name)  // intentional double-release
        let count = await lease.count(for: name)
        #expect(count == 0)
    }

    @Test func activeNames_only_includes_held_leases() async {
        let lease = ModelLease.shared
        let name = "active-\(UUID().uuidString)"
        let unrelated = "unrelated-\(UUID().uuidString)"

        await lease.acquire(name)
        let active = await lease.activeNames()
        #expect(active.contains(name))
        #expect(!active.contains(unrelated))

        await lease.release(name)
        let activeAfter = await lease.activeNames()
        #expect(!activeAfter.contains(name))
    }

    /// Simulates the shape of the fix's cancellation chain:
    ///
    ///   caller Task acquires lease → work stalls → caller cancelled
    ///   → `withTaskCancellationHandler.onCancel` + `defer` release the
    ///   lease → another waiter on `waitForZero` returns promptly.
    ///
    /// This is the "HTTPHandler cancel-on-close → ModelRuntime lease drop"
    /// contract, but scoped to the lease layer where it's fast and
    /// deterministic to test.
    @Test
    func cancelling_holder_drops_lease_and_unblocks_waiter() async {
        let lease = ModelLease.shared
        let name = "cancel-chain-\(UUID().uuidString)"

        let holderStarted = AtomicBoolFlag()
        let holder = Task {
            await lease.acquire(name)
            holderStarted.set()
            await withTaskCancellationHandler {
                while !Task.isCancelled {
                    try? await Task.sleep(nanoseconds: 50_000_000)
                }
            } onCancel: {
                // Fire-and-forget; matches `ModelRuntime.generateEventStream`'s
                // wrapper task which releases inside an async Task<Void,Never>.
                Task { await lease.release(name) }
            }
        }

        // Wait for the holder to actually acquire.
        while !holderStarted.value {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(await lease.count(for: name) == 1)

        // Kick off a waiter that mirrors `unload → waitForZero(name)`.
        let waiterDone = AtomicBoolFlag()
        let waiter = Task {
            await lease.waitForZero(name)
            waiterDone.set()
        }

        // Holder still running → waiter blocked.
        try? await Task.sleep(nanoseconds: 50_000_000)
        #expect(!waiterDone.value)

        // Cancel the holder. Without the onCancel branch the lease would
        // stay at 1 forever and the waiter would hang.
        holder.cancel()
        _ = await holder.value

        // Waiter must complete quickly once the cancel-driven release
        // lands. 2s is generous; in practice this is sub-ms.
        let deadline = Date().addingTimeInterval(2.0)
        while !waiterDone.value && Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        await waiter.value
        #expect(waiterDone.value)
        #expect(await lease.count(for: name) == 0)
    }
}

private final class AtomicBoolFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = false
    var value: Bool { lock.withLock { _value } }
    func set() { lock.withLock { _value = true } }
}
