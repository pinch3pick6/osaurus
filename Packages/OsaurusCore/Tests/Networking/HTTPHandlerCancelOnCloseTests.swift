//
//  HTTPHandlerCancelOnCloseTests.swift
//  osaurusTests
//
//  Verifies that when an HTTP client disconnects mid-request the server's
//  per-request `Task` is cancelled promptly. This is the plumbing that
//  makes `ModelRuntime`'s per-stream `ModelLease` drop on disconnect, so
//  that `strictSingleModel` eviction for the next request doesn't block
//  indefinitely on a zombie producer (the original bug the fix addresses).
//
//  We use a raw POSIX socket rather than URLSession so the disconnect is
//  deterministic: `Darwin.close(fd)` on the client-side socket immediately
//  sends FIN + RST to the server, which triggers `channelInactive` and
//  `channel.closeFuture` on the NIO side. URLSession's `cancel()` is
//  asynchronous and offers no such guarantees — it made this test flaky
//  in CI, which defeats its purpose.
//
//  The cancellation chain on the production path is:
//
//      channel.closeFuture → launchRequestTask's Task.cancel()
//        → `for try await delta in stream` throws CancellationError
//        → ChatEngine's stream's onTermination fires
//        → ModelRuntime's stream's onTermination fires
//        → activeGenerationTask's withTaskCancellationHandler releases lease
//
//  The test asserts the first two links (the HTTP Task observes cancel,
//  and the engine's producer task sees `Task.isCancelled`). The lease
//  release itself is covered in `ModelLeaseTests`.
//

import Darwin
import Foundation
import NIOCore
import NIOHTTP1
import NIOPosix
import Testing

@testable import OsaurusCore

@Suite(.serialized)
struct HTTPHandlerCancelOnCloseTests {

    /// SSE stream — client slams the socket mid-stream, server-side stream
    /// Task must observe cancellation. Without the `channel.closeFuture →
    /// task.cancel()` wiring the engine would never see `Task.isCancelled`
    /// and the producer would spin forever holding its (real) `ModelLease`.
    @Test
    func client_disconnect_cancels_sse_request_task() async throws {
        let engine = BlockingStreamEngine()
        let server = try await startTestServer(with: engine)

        let path = "/chat/completions"
        let headers = ["Accept: text/event-stream"]
        let body = try makeChatRequestBody(stream: true)

        let fd = try openPost(host: server.host, port: server.port, path: path,
                              extraHeaders: headers, body: body)

        // Let the server actually start the request. We don't care about
        // received bytes, only about server state; `BlockingStreamEngine`
        // sets `started` as soon as its producer Task runs.
        try await waitForTrue(timeout: 2.0) { engine.started.value }

        // Slam the TCP connection shut. This fires channelInactive on
        // the server, which runs through `channel.closeFuture` and
        // cancels the request Task (and, via channelInactive itself,
        // also clears the `currentTaskRef` slot as a safety net).
        _ = Darwin.shutdown(fd, SHUT_RDWR)
        _ = Darwin.close(fd)

        try await waitForTrue(timeout: 3.0) { engine.cancelled.value }
        try await waitForTrue(timeout: 1.0) { engine.producerExited.value }

        await server.shutdown()
    }

    /// NDJSON streaming path — same invariant. The `/chat` endpoint uses
    /// `streamChat` under the hood (unlike the non-streaming JSON branch
    /// of `/v1/chat/completions`), so a client disconnect here should also
    /// propagate down through `launchRequestTask` and cancel the engine.
    @Test
    func client_disconnect_cancels_ndjson_request_task() async throws {
        let engine = BlockingStreamEngine()
        let server = try await startTestServer(with: engine)

        // `stream: true` flag in the body isn't strictly required for the
        // /chat (NDJSON) route since that route is always streaming, but
        // keep it on for clarity.
        let body = try makeChatRequestBody(stream: true)
        let fd = try openPost(host: server.host, port: server.port, path: "/chat",
                              extraHeaders: [], body: body)

        try await waitForTrue(timeout: 2.0) { engine.started.value }
        _ = Darwin.shutdown(fd, SHUT_RDWR)
        _ = Darwin.close(fd)

        try await waitForTrue(timeout: 3.0) { engine.cancelled.value }
        try await waitForTrue(timeout: 1.0) { engine.producerExited.value }

        await server.shutdown()
    }
}

// MARK: - Raw socket helpers

private enum SocketError: Error { case connectFailed(Int32); case sendFailed(Int32) }

/// Open a TCP socket to `host:port`, send a fully-formed POST (headers +
/// body), and return the raw file descriptor. The caller is responsible
/// for `close(fd)`. We keep the socket in blocking mode — the test never
/// reads from it (the point is to trigger the server and then slam the
/// connection shut, not to parse the response).
private func openPost(
    host: String,
    port: Int,
    path: String,
    extraHeaders: [String],
    body: Data
) throws -> Int32 {
    let fd = Darwin.socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)
    guard fd >= 0 else { throw SocketError.connectFailed(errno) }

    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = in_port_t(port).bigEndian
    _ = host.withCString { inet_pton(AF_INET, $0, &addr.sin_addr) }

    let connectOK = withUnsafePointer(to: &addr) { ptr -> Int32 in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockaddrPtr in
            Darwin.connect(fd, sockaddrPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard connectOK == 0 else {
        let e = errno
        Darwin.close(fd)
        throw SocketError.connectFailed(e)
    }

    var headers = [
        "POST \(path) HTTP/1.1",
        "Host: \(host):\(port)",
        "Content-Type: application/json",
        "Content-Length: \(body.count)",
        "Authorization: Bearer \(TestAuth.bearerToken)",
        "Connection: close",
    ]
    headers.append(contentsOf: extraHeaders)
    var request = Data()
    request.append(Data((headers.joined(separator: "\r\n") + "\r\n\r\n").utf8))
    request.append(body)

    try request.withUnsafeBytes { buf -> Void in
        var sent = 0
        while sent < buf.count {
            let n = buf.baseAddress!.advanced(by: sent)
            let wrote = Darwin.send(fd, n, buf.count - sent, 0)
            if wrote <= 0 {
                let e = errno
                Darwin.close(fd)
                throw SocketError.sendFailed(e)
            }
            sent += wrote
        }
    }
    return fd
}

private func makeChatRequestBody(stream: Bool) throws -> Data {
    let req = ChatCompletionRequest(
        model: "fake",
        messages: [ChatMessage(role: "user", content: "hi")],
        temperature: 0.0,
        max_tokens: 16,
        stream: stream,
        top_p: nil,
        frequency_penalty: nil,
        presence_penalty: nil,
        stop: nil,
        n: nil,
        tools: nil,
        tool_choice: nil,
        session_id: nil
    )
    return try JSONEncoder().encode(req)
}

// MARK: - Test helpers

/// Poll `check` every 10ms until it returns true or the deadline passes.
/// A timeout records a test issue rather than hanging.
private func waitForTrue(
    timeout: TimeInterval,
    _ check: @Sendable () -> Bool
) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while Date() < deadline {
        if check() { return }
        try await Task.sleep(nanoseconds: 10_000_000)
    }
    Issue.record("waitForTrue timed out after \(timeout)s")
}

/// Minimal thread-safe flag.
private final class TestFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value = false
    var value: Bool { lock.withLock { _value } }
    func set() { lock.withLock { _value = true } }
}

/// Engine whose `streamChat` / `completeChat` both register "started",
/// suspend on a never-resolved sleep, record observed cancellation,
/// and set "producerExited" on the way out. Covers both streaming and
/// non-streaming HTTP paths from a single mock.
private final class BlockingStreamEngine: ChatEngineProtocol, @unchecked Sendable {
    let started = TestFlag()
    let cancelled = TestFlag()
    let producerExited = TestFlag()

    func streamChat(request: ChatCompletionRequest) async throws -> AsyncThrowingStream<
        String, Error
    > {
        let (stream, continuation) = AsyncThrowingStream<String, Error>.makeStream()
        let started = self.started
        let cancelled = self.cancelled
        let producerExited = self.producerExited
        let producer = Task {
            started.set()
            // Trickle a token every ~50 ms. Any real model streams
            // continuously, and it's the continuous writes that let NIO
            // detect a peer FIN (via a failing write → `errorCaught` →
            // `context.close()` → `channelInactive`). A totally silent
            // producer doesn't exercise that path and also doesn't reflect
            // how real inference behaves.
            while !Task.isCancelled {
                continuation.yield(".")
                do {
                    try await Task.sleep(nanoseconds: 50_000_000)
                } catch {
                    break
                }
            }
            if Task.isCancelled { cancelled.set() }
            producerExited.set()
            continuation.finish()
        }
        continuation.onTermination = { @Sendable _ in producer.cancel() }
        return stream
    }

    /// Only streaming routes exercise this mock; `completeChat` isn't
    /// reached in these tests. The bug we're guarding against (dangling
    /// `ModelLease` across a client disconnect) only manifests on
    /// streaming paths, so we deliberately don't exercise the
    /// non-streaming path here — doing so would require periodic writes
    /// for the server to notice the disconnect, which is orthogonal to
    /// the cancellation plumbing under test.
    func completeChat(request: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        fatalError("completeChat not used in these tests")
    }
}

// MARK: - Test server bootstrap

private struct TestServer {
    let group: MultiThreadedEventLoopGroup
    let channel: Channel
    let host: String
    let port: Int

    func shutdown() async {
        _ = try? await channel.close()
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            group.shutdownGracefully { _ in cont.resume() }
        }
    }
}

@discardableResult
private func startTestServer(with engine: ChatEngineProtocol) async throws -> TestServer {
    let group = MultiThreadedEventLoopGroup(numberOfThreads: 1)
    let bootstrap = ServerBootstrap(group: group)
        .serverChannelOption(ChannelOptions.backlog, value: 256)
        .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
        .childChannelInitializer { channel in
            channel.pipeline.configureHTTPServerPipeline().flatMap {
                channel.pipeline.addHandler(
                    HTTPHandler(
                        configuration: .default,
                        apiKeyValidator: TestAuth.validator,
                        eventLoop: channel.eventLoop,
                        chatEngine: engine,
                        trustLoopback: false
                    )
                )
            }
        }
        .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
        .childChannelOption(ChannelOptions.socketOption(.tcp_nodelay), value: 1)
        .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
        .childChannelOption(ChannelOptions.recvAllocator, value: AdaptiveRecvByteBufferAllocator())

    let ch = try await bootstrap.bind(host: "127.0.0.1", port: 0).get()
    let addr = ch.localAddress
    let port = addr?.port ?? 0
    return TestServer(group: group, channel: ch, host: "127.0.0.1", port: port)
}
