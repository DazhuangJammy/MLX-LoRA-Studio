import Foundation
import Testing
@testable import MLXLoRAStudio

@Suite("Chat")
struct ChatTests {
    @Test("Thinking stays separate while streaming and after closing tag")
    func thinkingSegments() {
        let partial = ChatMessage.splitThinking("<think>考虑问题")
        #expect(partial.thinking == "考虑问题")
        #expect(partial.answer.isEmpty)
        let complete = ChatMessage.splitThinking("<think>考虑问题</think>\n\n这是回答")
        #expect(complete.thinking == "考虑问题")
        #expect(complete.answer == "这是回答")
        #expect(ChatMessage.splitThinking("普通回答").answer == "普通回答")
    }

    @Test("Chat appears between Live Metrics and Upload to HF")
    func sidebarOrder() throws {
        let visible = SidebarSection.allCases.filter { $0 != .synthetic && $0 != .about }
        let index = try #require(visible.firstIndex(of: .chat))
        #expect(visible[index - 1] == .metrics)
        #expect(visible[index + 1] == .upload)
    }

    @Test("Snapshots match their base model's LoRA catalog")
    @MainActor func identity() {
        #expect(ChatSession.modelIdentity("/tmp/models--Qwen--Qwen3-0.6B/snapshots/abc") == "Qwen/Qwen3-0.6B")
        #expect(ChatSession.modelIdentity("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B")
    }
    @Test("Generation errors restore the question instead of leaving consecutive user turns")
    @MainActor func failedTurnRecovery() {
        let chat = ChatSession(historyURL: nil)
        chat.messages = [ChatMessage(role: "user", raw: "old"), ChatMessage(role: "assistant", raw: "answer"),
                         ChatMessage(role: "user", raw: "retry this"), ChatMessage(role: "assistant", raw: "")]
        chat.isGenerating = true
        chat.receive(Data("{\"type\":\"error\",\"message\":\"Context full\"}\n".utf8))
        #expect(chat.messages.count == 2)
        #expect(chat.draft == "retry this")
        #expect(!chat.isGenerating)
    }

    @Test("Stopping before the first answer restores the prompt")
    @MainActor func emptyStopRecovery() {
        let chat = ChatSession(historyURL: nil)
        chat.messages = [ChatMessage(role: "user", raw: "old question"), ChatMessage(role: "assistant", raw: "old answer"),
                         ChatMessage(role: "user", raw: "question"), ChatMessage(role: "assistant", raw: "<think>")]
        chat.receive(Data("{\"type\":\"done\",\"stopped\":true,\"tokens\":0}\n".utf8))
        #expect(chat.messages.count == 2)
        #expect(chat.messages.last?.note == "")
        #expect(chat.messages.last?.raw == "old answer")
        #expect(chat.draft == "question")
    }

    @Test("Literal thinking tags in user text remain visible")
    func literalTags() {
        let message = ChatMessage(role: "user", raw: "What does <think> mean?")
        #expect(message.parts.thinking.isEmpty)
        #expect(message.parts.answer == message.raw)
    }

    @Test("Protocol buffers split multibyte characters until a complete event arrives")
    @MainActor func splitUTF8() {
        let chat = ChatSession(historyURL: nil)
        chat.messages = [ChatMessage(role: "assistant", raw: "")]
        let data = Data("{\"type\":\"delta\",\"text\":\"你好\"}\n".utf8)
        let index = data.firstIndex(of: 0xE4)!
        chat.receive(data.prefix(index + 1))
        #expect(chat.messages[0].raw.isEmpty)
        chat.receive(data.suffix(from: index + 1))
        #expect(chat.messages[0].raw == "你好")
    }
    @Test("Conversation survives unload and an application restart; only clear deletes it")
    @MainActor func persistentHistory() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appendingPathComponent("history.json")
        let original = ChatSession(historyURL: file)
        original.messages = [ChatMessage(role: "user", raw: "记住暗号：蓝鲸"),
                             ChatMessage(role: "assistant", raw: "记住了。")]
        original.draft = "暗号是什么？"
        original.unload()
        #expect(original.messages.count == 2)
        let restored = ChatSession(historyURL: file)
        #expect(restored.messages.map(\.raw) == original.messages.map(\.raw))
        #expect(restored.messages.first?.id == original.messages.first?.id)
        #expect(restored.draft == "暗号是什么？")
        restored.clear()
        let cleared = ChatSession(historyURL: file)
        #expect(cleared.messages.isEmpty)
        #expect(cleared.draft.isEmpty)
    }

    @Test("Failed model reload does not erase saved conversation")
    @MainActor func failedLoadKeepsHistory() {
        let chat = ChatSession(historyURL: nil)
        chat.messages = [ChatMessage(role: "user", raw: "question"), ChatMessage(role: "assistant", raw: "answer")]
        chat.unload()
        chat.receive(Data("{\"type\":\"error\",\"message\":\"Model missing\"}\n".utf8))
        #expect(chat.messages.count == 2)
        #expect(chat.messages[1].raw == "answer")
    }
    @Test("Unreadable history is preserved until explicitly cleared")
    @MainActor func unreadableHistory() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".json")
        defer { try? FileManager.default.removeItem(at: file) }
        try Data("broken archive".utf8).write(to: file)
        let chat = ChatSession(historyURL: file)
        #expect(chat.historyError != nil)
        chat.unload()
        #expect(try String(contentsOf: file, encoding: .utf8) == "broken archive")
        chat.clear()
        #expect(chat.historyError == nil)
        #expect(ChatSession(historyURL: file).messages.isEmpty)
    }
    @Test("Automatic LoRA menu retains only the latest three for the selected model")
    @MainActor func recentAdaptersOnly() {
        let chat = ChatSession(historyURL: nil)
        chat.model = "Qwen/Qwen3-0.6B"
        chat.adapters = [ChatAdapter(path: "/other", model: "other/model", name: "Other", quantization: nil)] +
            (0..<6).map { ChatAdapter(path: "/run-\($0)", model: "Qwen/Qwen3-0.6B", name: "run-\($0)", quantization: nil) }
        #expect(chat.matchingAdapters.map(\.name) == ["run-0", "run-1", "run-2"])
        chat.model = "other/model"
        #expect(chat.matchingAdapters.map(\.name) == ["Other"])
    }
}
