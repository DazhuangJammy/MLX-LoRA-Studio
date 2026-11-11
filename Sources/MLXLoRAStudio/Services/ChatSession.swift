import AppKit
import Foundation
import Observation

struct ChatMessage: Identifiable, Codable {
    let id: UUID
    let role: String
    var raw: String
    var note = ""

    init(role: String, raw: String, note: String = "", id: UUID = UUID()) {
        self.id = id; self.role = role; self.raw = raw; self.note = note
    }

    var parts: (thinking: String, answer: String) {
        role == "assistant" ? Self.splitThinking(raw) : ("", raw)
    }

    static func splitThinking(_ raw: String) -> (thinking: String, answer: String) {
        guard let start = raw.range(of: "<think>") else { return ("", raw) }
        let before = String(raw[..<start.lowerBound])
        let rest = raw[start.upperBound...]
        guard let end = rest.range(of: "</think>") else {
            return (String(rest).trimmingCharacters(in: .whitespacesAndNewlines), before)
        }
        return (String(rest[..<end.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines),
                (before + rest[end.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

struct ChatHistory: Codable {
    var messages: [ChatMessage]
    var draft: String
}

struct ChatAdapter: Identifiable {
    let path: String
    let model: String
    let name: String
    let quantization: [String: Any]?
    var id: String { path }
}

@MainActor
@Observable
final class ChatSession {
    var model = ""
    var adapter = ""
    var models: [String] = []
    var adapters: [ChatAdapter] = []
    var messages: [ChatMessage] = []
    var draft = ""
    var thinking = false
    var thinkingSupported = false
    var maxTokens = 1024
    var contextLimit = 4096
    var isLoading = false
    var isGenerating = false
    var isStopping = false
    var isLoaded = false
    var status = "Choose a model and load it to start chatting."
    var error: String?
    var loadedModel = ""
    var loadedAdapter = ""
    var loadedPath = ""
    var diagnostics = ""
    var promptTokens = 0
    var reusedTokens = 0
    var historyError: String?
    @ObservationIgnored private let historyURL: URL?
    @ObservationIgnored private var streamedChunks = 0
    @ObservationIgnored private var process: Process?
    @ObservationIgnored private var input: Pipe?
    @ObservationIgnored private var output: Pipe?
    @ObservationIgnored private var errorPipe: Pipe?
    @ObservationIgnored private var buffer = Data()
    @ObservationIgnored private var sessionID = UUID()
    @ObservationIgnored private var initialized = false
    @ObservationIgnored private var terminationObserver: NSObjectProtocol?

    nonisolated static var defaultHistoryURL: URL {
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(".mlxlorastudio/chat/history.json")
    }

    init(historyURL: URL? = ChatSession.defaultHistoryURL) {
        self.historyURL = historyURL
        if let historyURL, FileManager.default.fileExists(atPath: historyURL.path) {
            do {
                let saved = try JSONDecoder().decode(ChatHistory.self, from: Data(contentsOf: historyURL))
                messages = saved.messages
                draft = saved.draft
                recoverUnansweredTurn()
            } catch {
                historyError = "Could not restore saved chat: \(error.localizedDescription)"
            }
        }
        terminationObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.unload() }
        }
    }

    var busy: Bool { isLoading || isGenerating }
    var selectionMatches: Bool { model == loadedModel && adapter == loadedAdapter }
    var canSend: Bool { isLoaded && selectionMatches && !busy && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    // RunArchive supplies newest-first order. Keep the automatic menu compact.
    static let recentAdapterLimit = 3
    var matchingAdapters: [ChatAdapter] {
        Array(adapters.filter { Self.modelIdentity($0.model) == Self.modelIdentity(model) }.prefix(Self.recentAdapterLimit))
    }
    var outputLimit: Int { max(1, min(32768, contextLimit - 1)) }

    static func modelIdentity(_ path: String) -> String {
        let expanded = NSString(string: path).expandingTildeInPath
        if let component = expanded.split(separator: "/").first(where: { $0.hasPrefix("models--") }) {
            return component.dropFirst("models--".count).replacingOccurrences(of: "--", with: "/")
        }
        return expanded.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    func refresh(store: AppStore) {
        let cached = HFCacheScanner.scan().models.map(\.hfID)
        let runs = RunArchive.discoverPersistedRuns(outputRoot: store.outputRoot)
        adapters = runs.compactMap { run in
            guard let config = run.spec, config.modelFamily == .text else { return nil }
            let folder = run.folderURL.appendingPathComponent("adapters")
            guard FileManager.default.fileExists(atPath: folder.appendingPathComponent("adapters.safetensors").path),
                  FileManager.default.fileExists(atPath: folder.appendingPathComponent("adapter_config.json").path) else { return nil }
            let data = try? Data(contentsOf: run.folderURL.appendingPathComponent("run_spec.json"))
            let spec = data.flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] } ?? [:]
            var quant: [String: Any]?
            for (key, bits, group, mode) in [("load_in_4bits", 4, 128, "affine"), ("load_in_6bits", 6, 128, "affine"), ("load_in_8bits", 8, 128, "affine"), ("load_in_mxfp4", 4, 32, "mxfp4")] {
                if spec[key] as? Bool == true {
                    quant = ["bits": bits, "group_size": group, "mode": mode]
                    break
                }
            }
            return ChatAdapter(path: folder.path, model: config.model, name: run.id, quantization: quant)
        }
        var seen = Set<String>()
        models = (cached + store.customModelPaths + adapters.map(\.model)).filter { seen.insert($0).inserted }
        if !initialized {
            initialized = true
            model = models.first(where: { $0 == store.training.model }) ?? models.first ?? store.training.model
            adapter = matchingAdapters.first?.path ?? ""
        }
        if !busy && !adapter.isEmpty {
            let knownButOlder = adapters.contains { $0.path == adapter } && !matchingAdapters.contains { $0.path == adapter }
            if knownButOlder || !FileManager.default.fileExists(atPath: adapter) {
                adapter = matchingAdapters.first?.path ?? ""
            }
        }
        if !model.isEmpty && !models.contains(model) { models.append(model) }
    }

    func selectModel(_ value: String) {
        guard value != model else { return }
        model = value
        if !models.contains(value) { models.append(value) }
        adapter = matchingAdapters.first?.path ?? ""
    }

    func browseModel() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = HFCacheScanner.cacheRoot()
        panel.message = "Choose a model folder containing config.json, or a Hugging Face cache folder."
        if panel.runModal() == .OK, let url = panel.url { selectModel(url.path) }
    }

    func browseAdapter(outputRoot: String) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: NSString(string: outputRoot).expandingTildeInPath)
        panel.message = "Choose the adapters folder containing adapter_config.json and adapters.safetensors."
        if panel.runModal() == .OK, let url = panel.url { adapter = url.path }
    }

    func load(store: AppStore) {
        guard !busy else { return }
        if store.trainingRunner.isRunning {
            error = "Stop or finish training before loading Chat to free GPU memory."
            return
        }
        unload()
        error = nil
        diagnostics = ""
        isLoading = true
        status = "Starting chat worker…"
        let id = sessionID
        let modelChoice = model
        let adapterChoice = adapter
        loadedModel = modelChoice
        loadedAdapter = adapterChoice
        let worker = URL(fileURLWithPath: store.workingDirectory).appendingPathComponent("Backend/chat_runner.py")
        let task = Process()
        task.executableURL = URL(fileURLWithPath: NSString(string: store.pythonExecutable).expandingTildeInPath)
        task.arguments = ["-u", worker.path]
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        HuggingFaceEnvironment.apply(store.huggingFaceToken(), to: &env)
        task.environment = env
        let stdin = Pipe(), stdout = Pipe(), stderr = Pipe()
        input = stdin; output = stdout; errorPipe = stderr
        task.standardInput = stdin; task.standardOutput = stdout; task.standardError = stderr
        stdout.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            Task { @MainActor [weak self] in
                guard let self, self.sessionID == id else { return }
                self.receive(data)
            }
        }
        stderr.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            let text = String(decoding: data, as: UTF8.self)
            Task { @MainActor [weak self] in
                guard let self, self.sessionID == id else { return }
                self.diagnostics = String((self.diagnostics + text).suffix(12000))
            }
        }
        task.terminationHandler = { [weak self] task in
            let code = task.terminationStatus
            Task { @MainActor [weak self] in
                guard let self, self.sessionID == id else { return }
                self.isLoading = false; self.isGenerating = false; self.isStopping = false; self.isLoaded = false
                self.recoverUnansweredTurn()
                self.saveHistory()
                self.status = "Chat worker stopped."
                self.error = "Chat worker exited (\(code)). Check diagnostics and the Python environment in Settings, then load again."
            }
        }
        process = task
        do {
            try task.run()
            var request: [String: Any] = ["command": "load", "model": modelChoice, "adapter": adapterChoice]
            if let quant = matchingAdapters.first?.quantization { request["quantization"] = quant }
            try write(request)
        } catch {
            unload()
            self.error = "Could not start Chat: \(error.localizedDescription). Check the Python environment in Settings."
        }
    }

    func unload() {
        if isGenerating {
            if messages.last?.role == "assistant" { messages[messages.count - 1].note = "Stopped" }
            recoverUnansweredTurn()
        }
        saveHistory()
        promptTokens = 0; reusedTokens = 0
        sessionID = UUID()
        output?.fileHandleForReading.readabilityHandler = nil
        errorPipe?.fileHandleForReading.readabilityHandler = nil
        try? input?.fileHandleForWriting.close()
        if let old = process, old.isRunning {
            old.terminate()
            Task {
                try? await Task.sleep(for: .seconds(2))
                if old.isRunning { kill(old.processIdentifier, SIGKILL) }
            }
        }
        process = nil; input = nil; output = nil; errorPipe = nil
        buffer = Data()
        isLoaded = false; isLoading = false; isGenerating = false; isStopping = false
        thinkingSupported = false
        status = "Model unloaded."
    }

    func send() {
        guard canSend else { return }
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        var history = messages.filter { !$0.raw.isEmpty }.map { ["role": $0.role, "content": $0.raw] }
        history.append(["role": "user", "content": text])
        do {
            try write(["command": "generate", "messages": history,
                       "thinking": thinking && thinkingSupported, "max_tokens": maxTokens])
            messages.append(ChatMessage(role: "user", raw: text))
            messages.append(ChatMessage(role: "assistant", raw: ""))
            draft = ""; error = nil; isGenerating = true; isStopping = false
            status = "Generating…"
            streamedChunks = 0
            saveHistory()
        } catch { self.error = error.localizedDescription }
    }

    func stop() {
        guard isGenerating else { return }
        do {
            try write(["command": "stop"])
            isStopping = true
            status = "Stopping after the current step…"
        } catch { self.error = error.localizedDescription; unload() }
    }

    func clear() {
        guard !busy else { return }
        if process?.isRunning == true {
            do { try write(["command": "clear_cache"]) }
            catch { unload() }
        }
        messages = []; draft = ""; error = nil; historyError = nil
        promptTokens = 0; reusedTokens = 0
        saveHistory()
        status = isLoaded ? "Ready · Chat and context cleared" : "Chat cleared. Load a model to start."
    }

    func saveHistory() {
        guard let historyURL else { return }
        // Preserve an unreadable archive until the user explicitly clears it.
        guard historyError == nil || !messages.isEmpty else { return }
        do {
            try FileManager.default.createDirectory(at: historyURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let data = try JSONEncoder().encode(ChatHistory(messages: messages, draft: draft))
            try data.write(to: historyURL, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: historyURL.path)
            historyError = nil
        } catch {
            historyError = "Chat could not be saved locally: \(error.localizedDescription)"
        }
    }

    private func write(_ request: [String: Any]) throws {
        guard let input, process?.isRunning == true else {
            throw NSError(domain: "Chat", code: 1, userInfo: [NSLocalizedDescriptionKey: "Load the model again; the chat worker is not running."])
        }
        var data = try JSONSerialization.data(withJSONObject: request)
        data.append(0x0A)
        try input.fileHandleForWriting.write(contentsOf: data)
    }

    private func recoverUnansweredTurn() {
        guard messages.count >= 2,
              let last = messages.last, last.role == "assistant",
              last.parts.answer.isEmpty, last.parts.thinking.isEmpty,
              messages[messages.count - 2].role == "user" else { return }
        let prompt = messages[messages.count - 2].raw
        messages.removeLast(2)
        if draft.isEmpty { draft = prompt }
    }

    func receive(_ data: Data) {
        buffer.append(data)
        while let end = buffer.firstIndex(of: 0x0A) {
            let line = buffer[..<end]
            buffer.removeSubrange(...end)
            guard let object = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { continue }
            switch object["type"] as? String {
            case "status": status = object["message"] as? String ?? status
            case "loaded":
                isLoading = false; isLoaded = true
                thinkingSupported = object["thinking_supported"] as? Bool ?? false
                contextLimit = object["context_limit"] as? Int ?? 4096
                maxTokens = min(maxTokens, outputLimit)
                loadedPath = object["model_path"] as? String ?? ""
                status = object["message"] as? String ?? "Ready"
            case "cache_cleared":
                reusedTokens = 0; promptTokens = 0
            case "started":
                promptTokens = object["prompt_tokens"] as? Int ?? 0
                reusedTokens = object["cached_tokens"] as? Int ?? 0
                if messages.last?.role == "assistant" { messages[messages.count - 1].raw = object["prefix"] as? String ?? "" }
            case "delta":
                if messages.last?.role == "assistant" { messages[messages.count - 1].raw += object["text"] as? String ?? "" }
                streamedChunks += 1
                if streamedChunks % 32 == 0 { saveHistory() }
            case "done":
                isGenerating = false; isStopping = false
                let stopped = object["stopped"] as? Bool ?? false
                let count = object["tokens"] as? Int ?? 0
                let capped = object["finish_reason"] as? String == "length"
                status = stopped ? "Stopped · \(count) tokens" : "Ready · \(count) tokens generated"
                if messages.last?.role == "assistant" {
                    messages[messages.count - 1].note = stopped ? "Stopped" : (capped ? "Output limit reached. Increase Max output to allow a longer reply." : "")
                }
                if stopped { recoverUnansweredTurn() }
                saveHistory()
            case "error":
                error = object["message"] as? String ?? "Unknown chat error"
                isLoading = false; isGenerating = false; isStopping = false
                status = isLoaded ? "Ready · Last request failed" : "Load failed"
                recoverUnansweredTurn()
                saveHistory()
            default: break
            }
        }
    }
}
