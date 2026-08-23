//! In-memory fakes for the harness layers the scheduler path touches.
//!
//! They play the role of the durable exoharness service plus the LLM-backed
//! conversation, so the system-under-test is exactly the scheduler + adapter
//! store/runtime code, with no subprocesses and no external network. State
//! lives in memory and survives simulated scheduler-runner crashes (the
//! service is a different process in production), except where a method is
//! wired to a buggify site to inject the faults the real service can produce.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::anyhow;
use async_trait::async_trait;
use executor::{
    AgentConfig, AgentHarnessKind, AgentSandboxConfig, ConversationConfig, ConversationModelConfig,
    ExecutionStreamHandle, Harness, HarnessAgent, HarnessConversation, SandboxScope, SendRequest,
    SendResult,
};
use exoharness::{
    AgentHandle, AgentRecord, Artifact, ArtifactVersion, AttachSandboxRequest, Binding, BindingId,
    BindingRecord, CancelSandboxProcessRequest, CloseSandboxProcessInputRequest,
    ConversationHandle, ConversationId, ConversationRecord, CreateSandboxRequest, Event, EventId,
    EventQuery, ExoHarness, ForkSandboxRequest, ForkThreadRequest, GetEventsResult,
    GetSandboxProcessEventsResult, ListConversationsRequest, ListConversationsResult,
    NewConversationRequest, PutSecretRequest, ReadArtifactRequest, RestoreSandboxRequest, Result,
    RunInSandboxRequest, SandboxAttachment, SandboxId, SandboxProcess, SandboxProcessEventQuery,
    SandboxProcessParts, SandboxProcessRecord, SandboxProcessStatus, SandboxProvider,
    SandboxRecord, Secret, SecretId, SecretMetadata, SessionId, SnapshotId,
    StartSandboxProcessRequest, StartSandboxRequest, ThreadHandle, TurnHandle, TurnRecord, Uuid7,
    WaitSandboxProcessRequest, WriteArtifactRequest, WriteSandboxProcessInputRequest,
};
use futures::FutureExt;
use futures::io::Cursor;
use lingua::Message;

fn now_utc() -> chrono::DateTime<chrono::Utc> {
    // Virtual clock under Patina; deterministic.
    chrono::Utc::now()
}

/// One recorded wakeup delivery, as the conversation saw it.
#[derive(Debug, Clone)]
pub struct ReceivedWakeup {
    pub prompt: String,
}

/// Everything the fakes remember, shared across incarnations of the
/// scheduler-runner logic (the "service" survives runner crashes).
#[derive(Default)]
pub struct ServiceState {
    pub artifacts: HashMap<String, Vec<Vec<u8>>>, // path -> versions of contents
    pub artifact_ids: HashMap<String, Uuid7>,     // path -> stable artifact id
    pub sandboxes: Vec<String>,                   // created sandbox ids
    pub wakeups: Vec<ReceivedWakeup>,             // every prompt the conversation received
    pub send_failures_remaining: u32,             // scripted transient send failures
    pub commands_run: u64,                        // sandbox commands executed
    pub sends_inside: u32,                        // sends currently executing
    pub max_sends_inside: u32,                    // high-water mark of concurrent sends
}

pub type SharedState = Arc<Mutex<ServiceState>>;

/// One conversation send in progress; drop releases the slot (also on
/// cancellation, see `send`).
struct SendInFlight(SharedState);

impl SendInFlight {
    fn enter(state: &SharedState) -> Self {
        let mut guard = state.lock().expect("service state poisoned");
        guard.sends_inside += 1;
        guard.max_sends_inside = guard.max_sends_inside.max(guard.sends_inside);
        drop(guard);
        Self(Arc::clone(state))
    }
}

impl Drop for SendInFlight {
    fn drop(&mut self) {
        self.0.lock().expect("service state poisoned").sends_inside -= 1;
    }
}

/// Scripted behavior for one sandbox command, drawn by the driver up front so
/// the workload is a pure function of the run seed.
#[derive(Debug, Clone)]
pub struct CommandScript {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
    /// Virtual-time cost of the command before it exits.
    pub run_millis: u64,
}

impl Default for CommandScript {
    fn default() -> Self {
        Self {
            stdout: b"ok".to_vec(),
            stderr: Vec::new(),
            exit_code: 0,
            run_millis: 50,
        }
    }
}

pub struct FakeSandboxProcess {
    script: CommandScript,
}

impl SandboxProcess for FakeSandboxProcess {
    fn into_parts(self: Box<Self>) -> SandboxProcessParts {
        let millis = self.script.run_millis;
        let exit = self.script.exit_code;
        SandboxProcessParts {
            stdout: Box::pin(Cursor::new(self.script.stdout)),
            stderr: Box::pin(Cursor::new(self.script.stderr)),
            stdin: Box::pin(Cursor::new(Vec::new())),
            wait: async move {
                tokio::time::sleep(std::time::Duration::from_millis(millis)).await;
                Ok(exit)
            }
            .boxed(),
        }
    }
}

pub struct FakeHarness {
    pub agent: Arc<FakeAgent>,
}

impl FakeHarness {
    pub fn new(state: SharedState, scripts: Vec<CommandScript>) -> Arc<Self> {
        let agent_id: Uuid7 = "00000000-0000-7000-8000-0000000000aa"
            .parse()
            .expect("fixed agent uuid");
        let conversation_id: Uuid7 = "00000000-0000-7000-8000-0000000000cc"
            .parse()
            .expect("fixed conversation uuid");
        let conversation = Arc::new(FakeConversation {
            record: ConversationRecord {
                id: conversation_id,
                slug: "conversation".to_string(),
                name: "Conversation".to_string(),
                latest_event_id: None,
            },
            state: Arc::clone(&state),
            scripts: Mutex::new(scripts.clone()),
        });
        let agent = Arc::new(FakeAgent {
            record: AgentRecord {
                id: agent_id,
                slug: "agent".to_string(),
                name: "Agent".to_string(),
            },
            state,
            conversation,
            scripts: Mutex::new(scripts),
        });
        Arc::new(Self { agent })
    }

    pub fn agent_id(&self) -> String {
        self.agent.record.id.to_string()
    }

    pub fn conversation_id(&self) -> String {
        self.agent.conversation.record.id.to_string()
    }
}

#[async_trait]
impl Harness for FakeHarness {
    fn exoharness_handle(&self) -> Arc<dyn ExoHarness> {
        unimplemented!("scheduler path does not use the root exoharness handle")
    }

    async fn list_agents(&self) -> Result<Vec<AgentRecord>> {
        Ok(vec![self.agent.record.clone()])
    }

    async fn get_agent(&self, agent_ref: &str) -> Result<Option<Arc<dyn HarnessAgent>>> {
        if patina_dst::buggify!("harness-agent-lookup-fails") {
            return Err(anyhow!("transient: agent lookup failed").into());
        }
        if agent_ref == self.agent.record.id.to_string() || agent_ref == self.agent.record.slug {
            Ok(Some(Arc::clone(&self.agent) as Arc<dyn HarnessAgent>))
        } else {
            Ok(None)
        }
    }

    async fn create_agent(
        &self,
        _request: executor::CreateAgentRequest,
    ) -> Result<Arc<dyn HarnessAgent>> {
        unimplemented!("driver creates no agents")
    }

    async fn delete_agent(&self, _agent_ref: &str) -> Result<bool> {
        unimplemented!("driver deletes no agents")
    }

    async fn flush_tracing(&self) -> Result<()> {
        Ok(())
    }
}

pub struct FakeAgent {
    pub record: AgentRecord,
    pub state: SharedState,
    pub conversation: Arc<FakeConversation>,
    scripts: Mutex<Vec<CommandScript>>,
}

#[async_trait]
impl HarnessAgent for FakeAgent {
    fn record(&self) -> &AgentRecord {
        &self.record
    }

    fn exoharness_handle(&self) -> Arc<dyn AgentHandle> {
        Arc::new(FakeAgentHandle {
            record: self.record.clone(),
            state: Arc::clone(&self.state),
            scripts_source: next_script_fn(&self.scripts, &self.state),
        })
    }

    async fn config(&self) -> Result<AgentConfig> {
        Ok(fake_agent_config())
    }

    async fn put_config(&self, _config: AgentConfig) -> Result<()> {
        unimplemented!()
    }

    async fn list_conversations(&self) -> Result<Vec<ConversationRecord>> {
        Ok(vec![self.conversation.record.clone()])
    }

    async fn get_conversation(
        &self,
        conversation_ref: &str,
    ) -> Result<Option<Arc<dyn HarnessConversation>>> {
        if conversation_ref == self.conversation.record.id.to_string()
            || conversation_ref == self.conversation.record.slug
        {
            Ok(Some(
                Arc::clone(&self.conversation) as Arc<dyn HarnessConversation>
            ))
        } else {
            Ok(None)
        }
    }

    async fn create_conversation(
        &self,
        _request: executor::CreateConversationRequest,
    ) -> Result<Arc<dyn HarnessConversation>> {
        unimplemented!()
    }

    async fn delete_conversation(&self, _conversation_ref: &str) -> Result<bool> {
        unimplemented!()
    }
}

pub fn fake_agent_config() -> AgentConfig {
    AgentConfig {
        instructions: vec![],
        harness: AgentHarnessKind::Basic,
        typescript: None,
        enable_agent_tool_creation: false,
        sandbox: AgentSandboxConfig {
            scope: SandboxScope::Agent,
            image: Some("fake-image".to_string()),
            provider: SandboxProvider::LocalProcess,
            mounts: vec![],
            enable_networking: false,
        },
        model: "fake-model".to_string(),
        max_output_tokens: None,
        max_tool_round_trips: Some(4),
        braintrust: None,
    }
}

pub struct FakeConversation {
    pub record: ConversationRecord,
    pub state: SharedState,
    scripts: Mutex<Vec<CommandScript>>,
}

#[async_trait]
impl HarnessConversation for FakeConversation {
    fn record(&self) -> &ConversationRecord {
        &self.record
    }

    fn exoharness_handle(&self) -> Arc<dyn ConversationHandle> {
        Arc::new(FakeThreadHandle {
            record: self.record.clone(),
            state: Arc::clone(&self.state),
            scripts_source: next_script_fn(&self.scripts, &self.state),
        })
    }

    async fn config(&self) -> Result<ConversationConfig> {
        Ok(ConversationConfig {
            sandbox_image: None,
            sandbox_provider: None,
            shell_program: None,
            mounts: vec![],
            durable_file_systems: vec![],
            sandbox_scope: None,
        })
    }

    async fn put_config(&self, _config: ConversationConfig) -> Result<()> {
        unimplemented!()
    }

    async fn model_override(&self) -> Result<Option<ConversationModelConfig>> {
        Ok(None)
    }

    async fn put_model_override(&self, _config: Option<ConversationModelConfig>) -> Result<()> {
        unimplemented!()
    }

    async fn messages(&self) -> Result<Vec<Message>> {
        Ok(vec![])
    }

    async fn close_session(&self, _session_id: SessionId) -> Result<()> {
        Ok(())
    }

    async fn send(&self, request: SendRequest) -> Result<SendResult> {
        // Track concurrent entries: the wakeup file lock exists precisely to
        // keep two wakeup turns off one conversation at the same time. The
        // entry is released by a drop guard, not a trailing statement: the
        // driver's power cuts cancel a scheduler pass at an arbitrary await
        // point, which drops in-flight sends — a send that never reaches its
        // decrement would leak a phantom "inside" count and turn every later
        // (properly serialized) send into a false double-hold.
        let _inside = SendInFlight::enter(&self.state);
        // Model a conversation turn: it costs virtual time and can fail
        // transiently, exactly like a real model call.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        let result = {
            let mut state = self.state.lock().expect("service state poisoned");
            if state.send_failures_remaining > 0 || patina_dst::buggify!("conversation-send-fails")
            {
                state.send_failures_remaining = state.send_failures_remaining.saturating_sub(1);
                patina_dst::sometimes!(true, "wakeup-send-failed");
                Err::<(), exoharness::Error>(anyhow!("transient: conversation send failed").into())
            } else {
                let prompt = request
                    .input
                    .iter()
                    .map(render_message_text)
                    .collect::<Vec<_>>()
                    .join("\n");
                state.wakeups.push(ReceivedWakeup { prompt });
                Ok(())
            }
        };
        result?;
        Ok(SendResult {
            session_id: Uuid7::now(),
            turn_id: Uuid7::now(),
            latest_event_id: Uuid7::now(),
        })
    }

    async fn send_stream(&self, _request: SendRequest) -> Result<ExecutionStreamHandle> {
        unimplemented!("scheduler wakeups use send, not send_stream")
    }
}

fn render_message_text(message: &Message) -> String {
    match message {
        Message::User { content } => match content {
            lingua::universal::UserContent::String(text) => text.clone(),
            other => format!("{other:?}"),
        },
        other => format!("{other:?}"),
    }
}

type ScriptFn = Box<dyn Fn() -> CommandScript + Send + Sync>;

/// Pops pre-drawn command scripts in order; falls back to the default once the
/// scripted ones are exhausted, so an unexpectedly chatty run stays defined.
fn next_script_fn(scripts: &Mutex<Vec<CommandScript>>, state: &SharedState) -> ScriptFn {
    let scripts: Vec<CommandScript> = scripts.lock().expect("scripts poisoned").clone();
    let state = Arc::clone(state);
    let cursor = Arc::new(Mutex::new(0usize));
    Box::new(move || {
        let mut index = cursor.lock().expect("script cursor poisoned");
        {
            let mut svc = state.lock().expect("service state poisoned");
            svc.commands_run += 1;
        }
        let script = scripts.get(*index).cloned().unwrap_or_default();
        *index += 1;
        script
    })
}

macro_rules! shared_sandbox_impl {
    ($ty:ident, $tag:literal) => {
        #[async_trait]
        impl exoharness::SandboxHandle for $ty {
            async fn list_sandboxes(&self) -> Result<Vec<SandboxRecord>> {
                Ok(vec![])
            }

            async fn create_sandbox(&self, request: CreateSandboxRequest) -> Result<SandboxId> {
                if patina_dst::buggify!(concat!("sandbox-create-fails-", $tag)) {
                    return Err(anyhow!("transient: sandbox create failed").into());
                }
                let id = request
                    .name
                    .unwrap_or_else(|| format!("sandbox-{}", Uuid7::now()));
                self.state
                    .lock()
                    .expect("service state poisoned")
                    .sandboxes
                    .push(id.clone());
                Ok(id)
            }

            async fn fork_sandbox(&self, _request: ForkSandboxRequest) -> Result<SandboxId> {
                unimplemented!()
            }

            async fn restore_sandbox(&self, _request: RestoreSandboxRequest) -> Result<SandboxId> {
                unimplemented!()
            }

            async fn terminate_sandbox(&self, _id: SandboxId) -> Result<()> {
                unimplemented!()
            }

            async fn attach_sandbox(&self, _request: AttachSandboxRequest) -> Result<SandboxId> {
                unimplemented!()
            }

            async fn detach_sandbox(&self, _id: SandboxId) -> Result<SandboxAttachment> {
                unimplemented!()
            }

            async fn stop_sandbox(&self, _id: SandboxId) -> Result<()> {
                unimplemented!()
            }

            async fn start_sandbox_process(
                &self,
                _request: StartSandboxProcessRequest,
            ) -> Result<SandboxProcessRecord> {
                unimplemented!()
            }

            async fn write_sandbox_process_input(
                &self,
                _request: WriteSandboxProcessInputRequest,
            ) -> Result<()> {
                unimplemented!()
            }

            async fn close_sandbox_process_input(
                &self,
                _request: CloseSandboxProcessInputRequest,
            ) -> Result<()> {
                unimplemented!()
            }

            async fn get_sandbox_process_events(
                &self,
                _query: SandboxProcessEventQuery,
            ) -> Result<GetSandboxProcessEventsResult> {
                unimplemented!()
            }

            async fn wait_sandbox_process(
                &self,
                _request: WaitSandboxProcessRequest,
            ) -> Result<SandboxProcessStatus> {
                unimplemented!()
            }

            async fn cancel_sandbox_process(
                &self,
                _request: CancelSandboxProcessRequest,
            ) -> Result<SandboxProcessStatus> {
                unimplemented!()
            }

            async fn run_in_sandbox(
                &self,
                _request: RunInSandboxRequest,
            ) -> Result<Box<dyn SandboxProcess>> {
                if patina_dst::buggify!(concat!("sandbox-run-fails-", $tag)) {
                    return Err(anyhow!("transient: run_in_sandbox failed").into());
                }
                Ok(Box::new(FakeSandboxProcess {
                    script: (self.scripts_source)(),
                }))
            }
        }
    };
}

macro_rules! shared_snapshot_impl {
    ($ty:ident) => {
        #[async_trait]
        impl exoharness::SnapshotHandle for $ty {
            async fn snapshot_sandbox(&self, _id: SandboxId) -> Result<SnapshotId> {
                unimplemented!()
            }

            async fn start_sandbox(&self, _request: StartSandboxRequest) -> Result<()> {
                unimplemented!()
            }
        }
    };
}

pub struct FakeAgentHandle {
    record: AgentRecord,
    state: SharedState,
    scripts_source: ScriptFn,
}

shared_snapshot_impl!(FakeAgentHandle);
shared_sandbox_impl!(FakeAgentHandle, "agent");

#[async_trait]
impl AgentHandle for FakeAgentHandle {
    fn record(&self) -> &AgentRecord {
        &self.record
    }

    async fn list_conversations(
        &self,
        _request: ListConversationsRequest,
    ) -> Result<ListConversationsResult<Arc<dyn ConversationHandle>>> {
        unimplemented!()
    }

    async fn get_conversation(
        &self,
        _id: &ConversationId,
    ) -> Result<Option<Arc<dyn ConversationHandle>>> {
        unimplemented!()
    }

    async fn new_conversation(
        &self,
        _request: NewConversationRequest,
    ) -> Result<Arc<dyn ConversationHandle>> {
        unimplemented!()
    }

    async fn delete_conversation(&self, _id: &ConversationId) -> Result<bool> {
        unimplemented!()
    }

    async fn list_bindings(&self) -> Result<Vec<BindingRecord>> {
        Ok(vec![])
    }

    async fn put_binding(&self, _binding: Binding) -> Result<BindingId> {
        unimplemented!()
    }

    async fn get_binding(&self, _id: &BindingId) -> Result<Option<Binding>> {
        Ok(None)
    }

    async fn list_secrets(&self) -> Result<Vec<SecretMetadata>> {
        Ok(vec![])
    }

    async fn put_secret(&self, _request: PutSecretRequest) -> Result<SecretId> {
        unimplemented!()
    }

    async fn get_secret(&self, _id: &SecretId) -> Result<Option<Secret>> {
        Ok(None)
    }

    async fn write_artifact(&self, request: WriteArtifactRequest) -> Result<ArtifactVersion> {
        if patina_dst::buggify!("artifact-write-fails-agent") {
            return Err(anyhow!("transient: artifact write failed").into());
        }
        let mut state = self.state.lock().expect("service state poisoned");
        let versions = state.artifacts.entry(request.path.clone()).or_default();
        versions.push(request.contents.clone());
        let version = versions.len() as u64;
        let artifact_id = *state
            .artifact_ids
            .entry(request.path.clone())
            .or_insert_with(Uuid7::now);
        Ok(ArtifactVersion {
            artifact_id,
            path: request.path,
            version,
            created_at: now_utc(),
            size_bytes: request.contents.len() as u64,
        })
    }

    async fn read_artifact(&self, request: ReadArtifactRequest) -> Result<Option<Artifact>> {
        let state = self.state.lock().expect("service state poisoned");
        let Some((path, _)) = state
            .artifact_ids
            .iter()
            .find(|(_, id)| **id == request.artifact_id)
        else {
            return Ok(None);
        };
        let versions = &state.artifacts[path];
        let version = request.version.unwrap_or(versions.len() as u64);
        let Some(contents) = versions.get((version as usize).saturating_sub(1)) else {
            return Ok(None);
        };
        Ok(Some(Artifact {
            version: ArtifactVersion {
                artifact_id: request.artifact_id,
                path: path.clone(),
                version,
                created_at: now_utc(),
                size_bytes: contents.len() as u64,
            },
            contents: contents.clone(),
        }))
    }

    async fn list_artifacts(&self) -> Result<Vec<ArtifactVersion>> {
        let state = self.state.lock().expect("service state poisoned");
        Ok(state
            .artifacts
            .iter()
            .map(|(path, versions)| ArtifactVersion {
                artifact_id: state.artifact_ids[path],
                path: path.clone(),
                version: versions.len() as u64,
                created_at: now_utc(),
                size_bytes: versions.last().map(Vec::len).unwrap_or(0) as u64,
            })
            .collect())
    }
}

pub struct FakeThreadHandle {
    record: ConversationRecord,
    state: SharedState,
    scripts_source: ScriptFn,
}

shared_snapshot_impl!(FakeThreadHandle);
shared_sandbox_impl!(FakeThreadHandle, "thread");

#[async_trait]
impl ThreadHandle for FakeThreadHandle {
    fn record(&self) -> &ConversationRecord {
        &self.record
    }

    async fn start_session(&self) -> Result<SessionId> {
        Ok(Uuid7::now())
    }

    async fn end_session(&self, _id: SessionId) -> Result<()> {
        Ok(())
    }

    async fn begin_turn(
        &self,
        _request: exoharness::BeginTurnRequest,
    ) -> Result<Arc<dyn TurnHandle>> {
        unimplemented!()
    }

    async fn turn_handle(&self, _record: TurnRecord) -> Result<Arc<dyn TurnHandle>> {
        unimplemented!()
    }

    async fn get_events(&self, _query: Option<EventQuery>) -> Result<GetEventsResult> {
        unimplemented!()
    }

    async fn watch_events(
        &self,
        _after_exclusive: std::ops::Bound<EventId>,
    ) -> Result<exoharness::EventStream> {
        unimplemented!()
    }

    async fn get_event(&self, _id: EventId) -> Result<Option<Event>> {
        unimplemented!()
    }

    async fn add_events(
        &self,
        _request: exoharness::AddEventsRequest,
    ) -> Result<exoharness::AddEventsResult> {
        unimplemented!()
    }

    async fn fork(&self, _request: ForkThreadRequest) -> Result<Arc<dyn ThreadHandle>> {
        unimplemented!()
    }

    async fn list_bindings(&self) -> Result<Vec<BindingRecord>> {
        Ok(vec![])
    }

    async fn put_binding(&self, _binding: Binding) -> Result<BindingId> {
        unimplemented!()
    }

    async fn get_binding(&self, _id: &BindingId) -> Result<Option<Binding>> {
        Ok(None)
    }

    async fn list_secrets(&self) -> Result<Vec<SecretMetadata>> {
        Ok(vec![])
    }

    async fn put_secret(&self, _request: PutSecretRequest) -> Result<SecretId> {
        unimplemented!()
    }

    async fn get_secret(&self, _id: &SecretId) -> Result<Option<Secret>> {
        Ok(None)
    }

    async fn write_artifact(&self, request: WriteArtifactRequest) -> Result<ArtifactVersion> {
        if patina_dst::buggify!("artifact-write-fails-thread") {
            return Err(anyhow!("transient: artifact write failed").into());
        }
        let mut state = self.state.lock().expect("service state poisoned");
        let versions = state.artifacts.entry(request.path.clone()).or_default();
        versions.push(request.contents.clone());
        let version = versions.len() as u64;
        let artifact_id = *state
            .artifact_ids
            .entry(request.path.clone())
            .or_insert_with(Uuid7::now);
        Ok(ArtifactVersion {
            artifact_id,
            path: request.path,
            version,
            created_at: now_utc(),
            size_bytes: request.contents.len() as u64,
        })
    }

    async fn read_artifact(&self, request: ReadArtifactRequest) -> Result<Option<Artifact>> {
        let state = self.state.lock().expect("service state poisoned");
        let Some((path, _)) = state
            .artifact_ids
            .iter()
            .find(|(_, id)| **id == request.artifact_id)
        else {
            return Ok(None);
        };
        let versions = &state.artifacts[path];
        let version = request.version.unwrap_or(versions.len() as u64);
        let Some(contents) = versions.get((version as usize).saturating_sub(1)) else {
            return Ok(None);
        };
        Ok(Some(Artifact {
            version: ArtifactVersion {
                artifact_id: request.artifact_id,
                path: path.clone(),
                version,
                created_at: now_utc(),
                size_bytes: contents.len() as u64,
            },
            contents: contents.clone(),
        }))
    }

    async fn list_artifacts(&self) -> Result<Vec<ArtifactVersion>> {
        let state = self.state.lock().expect("service state poisoned");
        Ok(state
            .artifacts
            .iter()
            .map(|(path, versions)| ArtifactVersion {
                artifact_id: state.artifact_ids[path],
                path: path.clone(),
                version: versions.len() as u64,
                created_at: now_utc(),
                size_bytes: versions.last().map(Vec::len).unwrap_or(0) as u64,
            })
            .collect())
    }
}
