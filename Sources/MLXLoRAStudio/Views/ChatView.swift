import SwiftUI

struct ChatView: View {
    let store: AppStore
    @Bindable var chat: ChatSession

    var body: some View {
        VStack(spacing: 0) {
            configuration
            Divider()
            conversation
            composer
        }
        .background(.background.opacity(0.45))
        .clipShape(RoundedRectangle(cornerRadius: 18))
        .padding(12)
        .navigationTitle("Chat")
        .task { chat.refresh(store: store) }
    }

    private var configuration: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text("Chat").font(.title2.bold())
                Text("Try your model, with or without LoRA.")
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Refresh") { chat.refresh(store: store) }
                    .disabled(chat.busy)
                    .help("Refresh downloaded models and completed training runs")
            }
            Grid(horizontalSpacing: 10, verticalSpacing: 10) {
                GridRow {
                    Text("Model").frame(width: 48, alignment: .leading)
                    Picker("Model", selection: Binding(get: { chat.model }, set: { chat.selectModel($0) })) {
                        if chat.models.isEmpty { Text("No downloaded models").tag(chat.model) }
                        ForEach(chat.models, id: \.self) { Text($0).tag($0) }
                    }
                    .labelsHidden()
                    .accessibilityLabel("Model")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .disabled(chat.busy)
                    Button("Browse…") { chat.browseModel() }.disabled(chat.busy)
                    Button(chat.isLoaded ? "Reload" : "Load model") { chat.load(store: store) }
                        .buttonStyle(.borderedProminent)
                        .disabled(chat.busy || chat.model.isEmpty || store.trainingRunner.isRunning)
                }
                GridRow {
                    Text("LoRA").frame(width: 48, alignment: .leading)
                    Picker("LoRA", selection: $chat.adapter) {
                        Text("None — base model").tag("")
                        ForEach(chat.matchingAdapters) { Text($0.name).tag($0.path) }
                        if !chat.adapter.isEmpty && !chat.matchingAdapters.contains(where: { $0.path == chat.adapter }) {
                            Text(URL(fileURLWithPath: chat.adapter).deletingLastPathComponent().lastPathComponent + " / adapters").tag(chat.adapter)
                        }
                    }
                    .labelsHidden()
                    .accessibilityLabel("LoRA")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .disabled(chat.busy)
                    Button("Browse…") { chat.browseAdapter(outputRoot: store.outputRoot) }.disabled(chat.busy)
                    Button(chat.isLoading ? "Cancel load" : "Unload") { chat.unload() }
                        .disabled(!chat.isLoaded && !chat.isLoading)
                }
            }
            HStack(spacing: 16) {
                Toggle("Thinking", isOn: $chat.thinking)
                    .toggleStyle(.switch)
                    .controlSize(.small)
                    .fixedSize()
                    .disabled(!chat.isLoaded || !chat.thinkingSupported || chat.busy)
                    .help(chat.thinkingSupported ? "Include the model’s thinking in a collapsible section" : "Available after loading a model whose template supports a thinking switch")
                HStack(spacing: 6) {
                    Text("Max output")
                    TextField("Tokens", value: $chat.maxTokens, format: .number.grouping(.never))
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 76)
                        .accessibilityLabel("Maximum output tokens")
                        .onChange(of: chat.maxTokens) { _, value in
                            chat.maxTokens = max(1, min(value, chat.outputLimit))
                        }
                    Text("tokens").foregroundStyle(.secondary)
                }
                .disabled(chat.busy)
                Spacer(minLength: 0)
                Button("Clear chat") { chat.clear() }
                    .disabled(chat.busy || (chat.messages.isEmpty && chat.draft.isEmpty && chat.historyError == nil))
            }
            Text(chat.isLoaded ? "Thinking and answer share the output limit. Context: \(chat.contextLimit.formatted()) tokens." : "Load a model to check thinking support and context size.")
                .font(.caption)
                .foregroundStyle(.secondary)
            if !chat.messages.isEmpty {
                Text("History: \(chat.messages.count) messages · Last prompt: \(chat.promptTokens.formatted()) tokens · Reused: \(chat.reusedTokens.formatted()) tokens")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let historyError = chat.historyError {
                Text(historyError).font(.caption).foregroundStyle(.red).textSelection(.enabled)
            }
            if chat.isLoaded && !chat.thinkingSupported {
                Text("This model’s template does not provide a thinking on/off switch.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            HStack(spacing: 8) {
                if chat.busy { ProgressView().controlSize(.small) }
                Text(chat.isLoaded && !chat.selectionMatches ? "Selection changed. Reload to chat with this model and LoRA." : chat.status)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                Spacer(minLength: 0)
            }
            if store.trainingRunner.isRunning {
                Text("Training is running. Finish or stop it before loading Chat.")
                    .font(.callout).foregroundStyle(.orange)
            }
            if let error = chat.error {
                VStack(alignment: .leading, spacing: 6) {
                    Text(error).foregroundStyle(.red).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                    if !chat.diagnostics.isEmpty {
                        DisclosureGroup("Diagnostics") {
                            ScrollView {
                                Text(chat.diagnostics).font(.caption.monospaced()).textSelection(.enabled)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }.frame(maxHeight: 100)
                        }
                    }
                }
                .font(.callout)
            }
        }
        .padding(20)
    }

    private var conversation: some View {
        ScrollViewReader { proxy in
            ScrollView {
                if chat.messages.isEmpty {
                    VStack(spacing: 10) {
                        Text("Ask your model a question").font(.title3.weight(.semibold))
                        Text("Load a model above, then send a message.\nChoose “None — base model” to compare it with your trained LoRA.")
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .frame(maxWidth: 440)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 52)
                    .padding(.horizontal, 24)
                } else {
                    LazyVStack(alignment: .leading, spacing: 24) {
                        ForEach(chat.messages) { message in
                            ChatMessageView(message: message, generating: chat.isGenerating && message.id == chat.messages.last?.id)
                        }
                        Color.clear.frame(height: 1).id("bottom")
                    }
                    .padding(24)
                    .frame(maxWidth: 860)
                    .frame(maxWidth: .infinity)
                }
            }
            .defaultScrollAnchor(.bottom)
            .onChange(of: chat.messages.count) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: chat.messages.last?.raw) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .bottom, spacing: 12) {
                ZStack(alignment: .topLeading) {
                    if chat.draft.isEmpty {
                        Text(chat.isLoaded ? "Message your model…" : "Load a model to start…")
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 5).padding(.top, 8)
                            .allowsHitTesting(false)
                    }
                    TextEditor(text: $chat.draft)
                        .font(.body)
                        .scrollContentBackground(.hidden)
                        .frame(height: 64)
                        .accessibilityLabel("Message")
                }
                if chat.isGenerating {
                    Button(chat.isStopping ? "Stopping…" : "Stop") { chat.stop() }
                        .disabled(chat.isStopping)
                } else {
                    Button("Send") { chat.send() }
                        .buttonStyle(.borderedProminent)
                        .disabled(!chat.canSend || store.trainingRunner.isRunning)
                }
            }
            .padding(10)
            .background(.background, in: RoundedRectangle(cornerRadius: 12))
            .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary))
            Text("⌘ Return to send · History is saved locally until you clear it")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 20).padding(.bottom, 16).padding(.top, 10)
    }
}

private struct ChatMessageView: View {
    let message: ChatMessage
    let generating: Bool
    @State private var showThinking = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(message.role == "user" ? "You" : "Assistant")
                    .font(.callout.weight(.semibold))
                    .foregroundStyle(message.role == "user" ? .secondary : .primary)
                Spacer()
                if !message.raw.isEmpty {
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(message.raw, forType: .string)
                    }.buttonStyle(.borderless).font(.caption)
                }
            }
            if !message.parts.thinking.isEmpty {
                DisclosureGroup(generating && message.parts.answer.isEmpty ? "Thinking…" : "Thinking", isExpanded: $showThinking) {
                    Text(message.parts.thinking).font(.callout)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 6)
                }
            }
            if !message.parts.answer.isEmpty {
                Text(.init(message.parts.answer))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if generating && message.parts.thinking.isEmpty {
                Text("Preparing response…").foregroundStyle(.secondary)
            }
            if !message.note.isEmpty {
                Text(message.note).font(.caption).foregroundStyle(.secondary)
            }
        }
        .padding(message.role == "user" ? 14 : 0)
        .background {
            if message.role == "user" {
                RoundedRectangle(cornerRadius: 12).fill(.quaternary.opacity(0.5))
            }
        }
    }
}
