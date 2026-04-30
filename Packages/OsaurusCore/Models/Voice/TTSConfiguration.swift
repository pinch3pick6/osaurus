//
//  TTSConfiguration.swift
//  osaurus
//
//  Configuration model for FluidAudio PocketTTS text-to-speech.
//

import Foundation

/// Configuration settings for PocketTTS text-to-speech.
public struct TTSConfiguration: Codable, Equatable, Sendable {
    /// Master enable toggle. When false, speaker buttons are hidden from message cells.
    public var enabled: Bool

    /// PocketTTS voice identifier.
    public var voice: String

    /// Generation temperature (0.1 – 1.2). Higher = more variation.
    public var temperature: Double

    public static let defaultVoice = "alba"

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = TTSConfiguration.default
        self.enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? defaults.enabled
        self.voice = try container.decodeIfPresent(String.self, forKey: .voice) ?? defaults.voice
        self.temperature =
            try container.decodeIfPresent(Double.self, forKey: .temperature) ?? defaults.temperature
    }

    public init(
        enabled: Bool = true,
        voice: String = TTSConfiguration.defaultVoice,
        temperature: Double = 0.7
    ) {
        self.enabled = enabled
        self.voice = voice
        self.temperature = temperature
    }

    public static var `default`: TTSConfiguration { TTSConfiguration() }
}

/// Handles persistence of `TTSConfiguration` with in-memory caching.
@MainActor
public enum TTSConfigurationStore {
    private static var cachedConfig: TTSConfiguration?

    public static func load() -> TTSConfiguration {
        if let cached = cachedConfig { return cached }
        let config = loadFromDisk()
        cachedConfig = config
        return config
    }

    public static func save(_ configuration: TTSConfiguration) {
        cachedConfig = configuration
        saveToDisk(configuration)
        NotificationCenter.default.post(name: .ttsConfigurationChanged, object: nil)
    }

    private static func loadFromDisk() -> TTSConfiguration {
        let url = OsaurusPaths.ttsConfigFile()
        guard FileManager.default.fileExists(atPath: url.path) else {
            return TTSConfiguration.default
        }
        do {
            return try JSONDecoder().decode(TTSConfiguration.self, from: Data(contentsOf: url))
        } catch {
            print("[Osaurus] Failed to load TTSConfiguration: \(error)")
            return TTSConfiguration.default
        }
    }

    private static func saveToDisk(_ configuration: TTSConfiguration) {
        let url = OsaurusPaths.ttsConfigFile()
        OsaurusPaths.ensureExistsSilent(url.deletingLastPathComponent())
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(configuration).write(to: url, options: [.atomic])
        } catch {
            print("[Osaurus] Failed to save TTSConfiguration: \(error)")
        }
    }
}

extension Notification.Name {
    public static let ttsConfigurationChanged = Notification.Name("osaurus.ttsConfigurationChanged")
}
