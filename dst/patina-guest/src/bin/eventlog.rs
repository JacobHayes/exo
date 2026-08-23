//! Deterministic-simulation driver for exo's exoharness storage layer
//! (`BasicExoHarness`): the append-only event log, agent/conversation CRUD,
//! and turn tracking — under Patina's virtual clock and in-memory fs, with
//! runner power cuts at arbitrary await points.
//!
//! Two phases:
//!  1. The crate's own `contract_tests` (storage-level ones only) run under
//!     the deterministic schedule — any panic is a contract breach under a
//!     schedule plain `cargo test` never produces.
//!  2. A crash-window scenario on the event log: append batches, cut some
//!     mid-flight, reopen the store cold, and check (a) every acknowledged
//!     event survived, and (b) the conversation record's `latest_event_id`
//!     head is not behind the events actually on disk (the record is written
//!     after the event files, so a cut in between strands the head).

use std::sync::Arc;

use anyhow::Result;
use exoharness::{
    AddEventsRequest, BasicExoHarness, BasicExoHarnessConfig, EventData, ExoHarness,
    NewAgentRequest, NewThreadRequest, SecretBackendChoice, Uuid7,
};
use patina_dst::VerdictKind;

const ROOT: &str = "/exo-dst/exoharness";

struct Prng(u64);

impl Prng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }

    fn chance(&mut self, permille: u64) -> bool {
        self.next() % 1000 < permille
    }
}

static VIOLATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn violation(label: &str, detail: &str) {
    VIOLATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    patina_dst::verdict(VerdictKind::Violation, label, detail);
}

fn config(root: &str) -> BasicExoHarnessConfig {
    BasicExoHarnessConfig {
        root: root.into(),
        secret_backend: SecretBackendChoice::Static([7u8; 32]),
        sandbox_default: exoharness::SandboxProvider::LocalProcess,
        // Registered but never exercised: no scenario here creates a
        // sandbox, so no subprocess is ever spawned.
        sandbox_backends: vec![exoharness::SandboxBackendRegistration::local_process()],
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut rng = Prng(patina_dst::rng());
    patina_dst::lifecycle::setup_complete();

    // ---- Phase 1: the crate's own storage contract under this schedule. ----
    let contract = Arc::new(BasicExoHarness::new(config("/exo-dst/contract")).await?);
    exoharness::contract_tests::supports_thread_api_and_conversation_compatibility(
        contract.clone(),
    )
    .await;
    exoharness::contract_tests::supports_agent_and_conversation_crud(contract.clone()).await;
    exoharness::contract_tests::list_conversations_returns_recent_first_and_paginates(
        contract.clone(),
    )
    .await;
    exoharness::contract_tests::begin_turn_tracks_events_through_finish(contract.clone()).await;
    patina_dst::reachable!("eventlog-contract-suite-passed");

    // ---- Phase 2: crash windows on the append path. ----
    let harness = Arc::new(BasicExoHarness::new(config(ROOT)).await?);
    let agent = harness
        .new_agent(NewAgentRequest {
            slug: "agent".into(),
            name: "Agent".into(),
        })
        .await?;
    let agent_id = agent.record().id;
    let conversation = agent
        .new_conversation(NewThreadRequest {
            slug: Some("dst".into()),
            name: None,
        })
        .await?;
    let conversation_id = conversation.record().id;

    let mut acked: Vec<u64> = Vec::new();
    let mut next_seq = 0u64;
    for _round in 0..14 {
        let batch: Vec<u64> = (0..rng.range(1, 4))
            .map(|_| {
                next_seq += 1;
                next_seq
            })
            .collect();
        let request = AddEventsRequest {
            session_id: None,
            turn_id: None,
            data: batch
                .iter()
                .map(|seq| EventData::Custom {
                    event_type: "dst-probe".to_string(),
                    payload: serde_json::json!({ "seq": seq }),
                })
                .collect(),
        };
        let append = conversation.add_events(request);
        if rng.chance(350) {
            let cut_after = rng.range(0, 8);
            tokio::select! {
                result = append => {
                    if result.is_ok() {
                        acked.extend(&batch);
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(cut_after)) => {
                    patina_dst::sometimes!(true, "eventlog-append-power-cut");
                }
            }
        } else if append.await.is_ok() {
            acked.extend(&batch);
        }
        tokio::time::sleep(std::time::Duration::from_millis(rng.range(1, 40))).await;
    }

    // Cold reopen: fresh in-memory state over the same on-disk root, the way a
    // restarted process would see it.
    drop((conversation, agent));
    let reopened = Arc::new(BasicExoHarness::new(config(ROOT)).await?);
    let agent = reopened
        .get_agent(&agent_id)
        .await?
        .expect("agent record must survive reopen");
    let conversation = agent
        .get_conversation(&conversation_id)
        .await?
        .expect("conversation record must survive reopen");

    let events = conversation.get_events(None).await?.events;
    let mut seen_seqs = Vec::new();
    let mut max_event_id: Option<Uuid7> = None;
    for event in &events {
        if let EventData::Custom {
            event_type,
            payload,
        } = &event.data
            && event_type == "dst-probe"
            && let Some(seq) = payload.get("seq").and_then(|value| value.as_u64())
        {
            seen_seqs.push(seq);
        }
        max_event_id = Some(max_event_id.map_or(event.id, |current| current.max(event.id)));
    }

    // (a) Durability of acknowledged appends.
    for seq in &acked {
        if !seen_seqs.contains(seq) {
            violation(
                "eventlog-acked-event-lost",
                &format!("event seq={seq} was acknowledged but is gone after reopen"),
            );
        }
    }
    patina_dst::sometimes!(
        seen_seqs.len() > acked.len(),
        "eventlog-uncommitted-tail-survived"
    );

    // (b) The record head must not be behind the events on disk: readers that
    // trust `latest_event_id` (head checks, pagination) are looking at a
    // truncated history even though `get_events` serves the full one.
    let head = conversation.record().latest_event_id;
    if let Some(max_id) = max_event_id {
        match head {
            Some(head) if head >= max_id => {}
            stale => violation(
                "eventlog-head-behind-events",
                &format!("record head {stale:?} but events on disk reach {max_id}"),
            ),
        }
    }

    let violations = VIOLATIONS.load(std::sync::atomic::Ordering::Relaxed);
    if violations > 0 {
        println!("EVENTLOG_DST_VIOLATIONS count={violations}");
        std::process::exit(1);
    }
    patina_dst::verdict(
        VerdictKind::Pass,
        "eventlog-outcome",
        &format!(
            "acked={} on_disk={} events_total={}",
            acked.len(),
            seen_seqs.len(),
            events.len()
        ),
    );
    println!(
        "EVENTLOG_DST_OK acked={} on_disk={}",
        acked.len(),
        seen_seqs.len()
    );
    Ok(())
}
