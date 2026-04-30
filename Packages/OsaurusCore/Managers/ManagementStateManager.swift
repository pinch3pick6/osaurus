//
//  ManagementStateManager.swift
//  osaurus
//
//  Manages the session state for the management interface.
//

import Foundation
import Combine

/// Manages the session state for the management interface.
@MainActor
public final class ManagementStateManager: ObservableObject {
    public static let shared = ManagementStateManager()

    /// Persists the last selected tab within the current app session.
    @Published public var selectedTab: ManagementTab = .settings

    /// One-shot request to focus a specific sub-tab inside `VoiceView`.
    /// VoiceView observes this and resets it to nil after applying.
    @Published public var voiceSubTabRequest: String?

    private init() {}
}
