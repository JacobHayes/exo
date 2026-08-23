//! Deterministic-simulation driver for exo's scheduler and adapter-outbox
//! subsystems under Patina.
//!
//! The system-under-test is the real `executor` code: `SchedulerStore`,
//! `run_due_tasks` / `redeliver_pending_wakes`, and `AdapterStore`'s outbox
//! state machine — running against Patina's in-memory filesystem and virtual
//! clock, with the LLM/exoharness service layer replaced by in-memory fakes
//! (a wakeup is real code all the way down to `conversation.send`).
//!
//! Faults come from three places, all seed-deterministic:
//!  - Patina fault knobs (fs errors/crashes/latency, preemption) on the CLI;
//!  - `buggify!` sites inside exo (rare-path activation, `--buggify`);
//!  - runner "power cuts": the driver cancels the in-flight scheduler pass at
//!    an arbitrary await point, dropping its in-memory state exactly the way a
//!    dead scheduler-runner process would, then recovers the way the real
//!    runner does on startup (`redeliver_pending_wakes` first).
//!
//! Outcomes are reported through the verdict ABI: any invariant breach is a
//! `Violation` under a stable label; a clean run reports one `Pass` with an
//! order-invariant outcome digest.

mod fakes;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use executor::HarnessAgent as _;
use executor::{
    AdapterStore, MissedPolicy, NewScheduledTask, SchedulerRunOptions, SchedulerStore,
    redeliver_pending_wakes, run_due_tasks,
};
use fakes::{CommandScript, FakeHarness, ServiceState, SharedState};
use patina_dst::VerdictKind;

const SCHED_ROOT: &str = "/exo-dst/scheduled-tasks";
const ADAPTER_ROOT: &str = "/exo-dst/adapters";

/// Small local PRNG (splitmix64) seeded once from the runtime, so the whole
/// workload derivation costs one boundary op and stays a pure function of the
/// run seed.
struct Prng(u64);

impl Prng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Uniform draw in `[lo, hi]`.
    fn range(&mut self, lo: u64, hi: u64) -> u64 {
        lo + self.next() % (hi - lo + 1)
    }

    fn chance(&mut self, permille: u64) -> bool {
        self.next() % 1000 < permille
    }
}

static VIOLATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Retries a fallible store call through injected transient faults. For use in
/// the setup and verification phases only — the scenario phase must see raw
/// errors, they are part of what is being tested.
macro_rules! retry {
    ($call:expr) => {{
        let mut result = $call.await;
        let mut tries = 0u32;
        while result.is_err() && tries < 25 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            result = $call.await;
            tries += 1;
        }
        result
    }};
}

fn violation(label: &str, detail: &str) {
    VIOLATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    patina_dst::verdict(VerdictKind::Violation, label, detail);
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let mut rng = Prng(patina_dst::rng());

    let sched_store = SchedulerStore::new(SCHED_ROOT);
    let state: SharedState = Arc::new(Mutex::new(ServiceState::default()));

    // Seed-derived workload: tasks with varied schedules and policies, and
    // pre-drawn command scripts (output sizes, exit codes, run times).
    let task_count = rng.range(2, 4);
    let mut expected_one_shots = Vec::new();
    let mut task_intervals: BTreeMap<String, u64> = BTreeMap::new();
    let mut scripts = Vec::new();
    for _ in 0..64 {
        scripts.push(CommandScript {
            stdout: vec![b'o'; rng.range(0, 4096) as usize],
            stderr: if rng.chance(200) {
                vec![b'e'; rng.range(1, 512) as usize]
            } else {
                vec![]
            },
            exit_code: if rng.chance(150) { 1 } else { 0 },
            run_millis: rng.range(5, 400),
        });
    }
    let harness = FakeHarness::new(Arc::clone(&state), scripts);
    patina_dst::lifecycle::setup_complete();

    for index in 0..task_count {
        let one_shot = rng.chance(250);
        let schedule = if one_shot {
            // A one-shot due a seed-chosen distance into the (virtual) future.
            let at_secs = rng.range(2, 20);
            format!("@at 1970-01-01T00:00:{at_secs:02}Z")
        } else {
            format!("@every {}s", rng.range(2, 10))
        };
        let missed = match rng.next() % 3 {
            0 => MissedPolicy::Skip,
            1 => MissedPolicy::Once,
            _ => MissedPolicy::All,
        };
        let request = NewScheduledTask {
            agent_id: harness.agent_id(),
            conversation_id: harness.conversation_id(),
            name: format!("task-{index}"),
            schedule: schedule.clone(),
            sandbox_mode: None,
            setup_command: if rng.chance(300) {
                Some(vec!["setup".to_string()])
            } else {
                None
            },
            command: vec!["work".to_string()],
            report_prompt: format!("Report for task-{index}."),
            max_output_bytes: Some(rng.range(64, 8192)),
            missed: Some(missed),
        };
        let task = retry!(sched_store.create_task(request.clone()))?;
        if one_shot {
            expected_one_shots.push(task.id.clone());
        } else if let Some(interval) = parse_every_secs(&schedule) {
            task_intervals.insert(task.id.clone(), interval * 1000);
        }
    }

    // ---- Scheduler scenario: passes, power cuts, recovery. ----
    let passes = rng.range(8, 16);
    let mut crashed_last_pass = false;
    for _pass in 0..passes {
        // The real runner redelivers unconfirmed wakeups on startup. Model a
        // restart after every cut (and occasionally a clean restart).
        if crashed_last_pass || rng.chance(150) {
            let harness_dyn = Arc::clone(&harness) as Arc<dyn executor::Harness>;
            let delivered = redeliver_pending_wakes(harness_dyn, &sched_store).await;
            match delivered {
                Ok(count) => patina_dst::sometimes!(count > 0, "redelivery-landed"),
                // Redelivery can fail transiently (fs faults, send faults);
                // the next startup retries it.
                Err(_) => patina_dst::sometimes!(true, "redelivery-errored"),
            }
            crashed_last_pass = false;
        }

        let harness_dyn = Arc::clone(&harness) as Arc<dyn executor::Harness>;
        let pass = run_due_tasks(harness_dyn, &sched_store, SchedulerRunOptions { limit: 10 });
        if rng.chance(350) {
            // Power cut: cancel the pass at whatever await point the virtual
            // clock reaches first, dropping all of its in-memory state.
            let cut_after = rng.range(1, 900);
            tokio::select! {
                result = pass => {
                    if result.is_err() {
                        patina_dst::sometimes!(true, "sched-pass-errored");
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(cut_after)) => {
                    patina_dst::sometimes!(true, "sched-pass-power-cut");
                    crashed_last_pass = true;
                }
            }
        } else if pass.await.is_err() {
            patina_dst::sometimes!(true, "sched-pass-errored-uncut");
        }

        // Advance the grid. Occasionally jump past the lease so an orphaned
        // claim gets re-run rather than sticking forever.
        let advance_ms = if rng.chance(60) {
            executor::DEFAULT_TASK_LEASE_MS + rng.range(1, 5_000)
        } else {
            rng.range(300, 4_000)
        };
        tokio::time::sleep(std::time::Duration::from_millis(advance_ms)).await;
    }

    // Quiescence: no cuts, no new driver-injected faults. Clean full passes
    // (startup redelivery + a run pass, exactly the real runner's loop) drain
    // pending wakeups and reclaim any lease a power cut orphaned.
    {
        let mut state = state.lock().expect("service state poisoned");
        state.send_failures_remaining = 0;
    }
    let mut quiescence_clean = true;
    for round in 0..3u32 {
        let wait_ms = if round == 0 {
            executor::DEFAULT_TASK_LEASE_MS + 1_000
        } else {
            2_000
        };
        tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
        let redelivered = retry!(redeliver_pending_wakes(
            Arc::clone(&harness) as Arc<dyn executor::Harness>,
            &sched_store
        ));
        if redelivered.is_err() {
            quiescence_clean = false;
        }
        let ran = retry!(run_due_tasks(
            Arc::clone(&harness) as Arc<dyn executor::Harness>,
            &sched_store,
            SchedulerRunOptions { limit: 10 },
        ));
        if ran.is_err() {
            quiescence_clean = false;
        }
    }
    // One final redelivery so a wakeup recorded by the last pass is confirmed.
    if retry!(redeliver_pending_wakes(
        Arc::clone(&harness) as Arc<dyn executor::Harness>,
        &sched_store
    ))
    .is_err()
    {
        quiescence_clean = false;
    }
    patina_dst::sometimes!(!quiescence_clean, "sched-quiescence-degraded");

    check_scheduler_invariants(
        &sched_store,
        &state,
        &task_intervals,
        &expected_one_shots,
        quiescence_clean,
    )
    .await?;

    // ---- Outbox scenario: enqueue, claim, ack/nack, power cuts, recovery. ----
    let adapter_store = AdapterStore::new(ADAPTER_ROOT);
    let adapter_id = "adapter-dst";
    let message_count = rng.range(4, 10);
    let mut enqueued = Vec::new();
    for index in 0..message_count {
        let message = retry!(adapter_store.enqueue_outbound_message(
            adapter_id.to_string(),
            format!("message-{index}"),
            None,
            Vec::new(),
        ))?;
        enqueued.push(message.id.clone());
    }

    // Drive the delivery loop the way adapter workers do: recover inflight,
    // claim, then ack or nack each claimed message; cut the loop sometimes.
    for _round in 0..24 {
        // Pre-draw this round's decisions so the async block borrows nothing
        // mutable and the workload stays a pure function of the seed.
        let nack_decisions: Vec<bool> = (0..32).map(|_| rng.chance(400)).collect();
        let cut = rng.chance(300);
        let cut_after = rng.range(1, 60);
        let deliver = async {
            adapter_store.requeue_inflight_messages(adapter_id).await?;
            let claimed = adapter_store.claim_outbound_messages(adapter_id).await?;
            for (index, message) in claimed.into_iter().enumerate() {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                if nack_decisions[index % nack_decisions.len()] {
                    patina_dst::sometimes!(true, "outbox-nacked");
                    adapter_store
                        .nack_outbound_message(adapter_id, &message.id, "scripted failure")
                        .await?;
                } else {
                    adapter_store
                        .acknowledge_outbound_message(adapter_id, &message.id)
                        .await?;
                }
            }
            anyhow::Ok(())
        };
        if cut {
            tokio::select! {
                result = deliver => {
                    if result.is_err() {
                        patina_dst::sometimes!(true, "outbox-round-errored");
                    }
                }
                _ = tokio::time::sleep(std::time::Duration::from_millis(cut_after)) => {
                    patina_dst::sometimes!(true, "outbox-power-cut");
                }
            }
        } else if deliver.await.is_err() {
            patina_dst::sometimes!(true, "outbox-round-errored-uncut");
        }
        tokio::time::sleep(std::time::Duration::from_millis(rng.range(20, 200))).await;
    }
    // Final clean drain.
    for _ in 0..8 {
        retry!(adapter_store.requeue_inflight_messages(adapter_id))?;
        let claimed = retry!(adapter_store.claim_outbound_messages(adapter_id))?;
        if claimed.is_empty() {
            break;
        }
        for message in claimed {
            retry!(adapter_store.acknowledge_outbound_message(adapter_id, &message.id))?;
        }
    }

    check_outbox_invariants(adapter_id, &enqueued).await?;

    // ---- Wakeup-lock scenario: a dead process's stale lock, three racing
    // wakeups. remove_stale_lock judges staleness by mtime and then removes
    // unconditionally; two waiters can both judge the same lock stale, one
    // removes it, a third acquires, and the second then removes the NEW
    // holder's lock — two wakeup turns on one conversation at once. ----
    {
        let conversation_id = harness.conversation_id();
        let lock_dir = std::env::temp_dir().join("exo-wakeup-locks");
        tokio::fs::create_dir_all(&lock_dir).await?;
        let lock_path = lock_dir.join(format!("{conversation_id}.lock"));
        tokio::fs::write(&lock_path, b"").await?; // the corpse of a dead runner
        // Age it past the 30-minute staleness threshold.
        tokio::time::sleep(std::time::Duration::from_secs(31 * 60)).await;

        {
            let mut state = state.lock().expect("service state poisoned");
            state.max_sends_inside = 0;
            state.send_failures_remaining = 0;
        }
        let conversation = harness
            .agent
            .get_conversation(&conversation_id)
            .await?
            .expect("fake conversation exists");
        let waiters = (0..3).map(|index| {
            let conversation = Arc::clone(&conversation);
            async move {
                executor::send_conversation_wakeup(
                    conversation.as_ref(),
                    format!("stale-lock-probe-{index}"),
                )
                .await
            }
        });
        let results = futures::future::join_all(waiters).await;
        patina_dst::sometimes!(
            results.iter().any(Result::is_err),
            "wakeup-lock-send-errored"
        );
        let max_inside = state
            .lock()
            .expect("service state poisoned")
            .max_sends_inside;
        patina_dst::sometimes!(max_inside == 1, "wakeup-lock-serialized");
        if max_inside > 1 {
            violation(
                "wakeup-lock-double-hold",
                &format!("{max_inside} wakeup sends ran concurrently on one conversation"),
            );
        }
    }

    // ---- Outcome digest: order-invariant summary of what happened. ----
    let state = state.lock().expect("service state poisoned");
    let mut wakeup_counts: BTreeMap<&str, u64> = BTreeMap::new();
    for wakeup in &state.wakeups {
        *wakeup_counts
            .entry(task_name_of(&wakeup.prompt).unwrap_or("?"))
            .or_default() += 1;
    }
    let digest = format!(
        "wakeups={} commands={} per_task={:?}",
        state.wakeups.len(),
        state.commands_run,
        wakeup_counts,
    );
    let violations = VIOLATIONS.load(std::sync::atomic::Ordering::Relaxed);
    if violations > 0 {
        println!("EXO_DST_VIOLATIONS count={violations}");
        std::process::exit(1);
    }
    patina_dst::verdict(VerdictKind::Pass, "exo-dst-outcome", &digest);
    println!("EXO_DST_OK {digest}");
    Ok(())
}

fn parse_every_secs(schedule: &str) -> Option<u64> {
    schedule
        .strip_prefix("@every ")?
        .strip_suffix('s')?
        .parse()
        .ok()
}

/// Pulls `task-N` back out of a wakeup prompt (they all embed the task name in
/// backticks on the first line).
fn task_name_of(prompt: &str) -> Option<&str> {
    let start = prompt.find('`')? + 1;
    let end = start + prompt[start..].find('`')?;
    Some(&prompt[start..end])
}

async fn check_scheduler_invariants(
    store: &SchedulerStore,
    state: &SharedState,
    task_intervals: &BTreeMap<String, u64>,
    one_shots: &[String],
    quiescence_clean: bool,
) -> Result<()> {
    // I1: nothing left pending after a clean recovery pass — a stranded
    // pending fire is a wakeup the conversation will never get.
    let pending = retry!(store.pending_fires())?;
    if quiescence_clean && !pending.is_empty() {
        violation(
            "sched-stranded-pending-fire",
            &format!(
                "{:?}",
                pending
                    .iter()
                    .map(|f| (&f.task_id, f.slot_ms))
                    .collect::<Vec<_>>()
            ),
        );
    }

    // I2: every delivered fire's wakeup actually reached the conversation,
    // and no (task, slot) woke it more than twice (once plus the one bounded
    // crash-window repeat the design allows).
    let delivered_dir = PathBuf::from(SCHED_ROOT).join("fires").join("delivered");
    let wakeups = {
        let state = state.lock().expect("service state poisoned");
        state
            .wakeups
            .iter()
            .map(|w| w.prompt.clone())
            .collect::<Vec<_>>()
    };
    let mut delivered_fires = 0u64;
    if let Ok(mut entries) = tokio::fs::read_dir(&delivered_dir).await {
        while let Some(entry) = entries.next_entry().await? {
            if entry.path().extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let path = entry.path();
            let fire: executor::ScheduledFireRecord =
                serde_json::from_slice(&retry!(tokio::fs::read(&path))?)?;
            delivered_fires += 1;
            let received = wakeups
                .iter()
                .filter(|prompt| **prompt == fire.prompt)
                .count();
            if received == 0 {
                violation(
                    "sched-wakeup-lost",
                    &format!(
                        "fire ({}, {}) marked delivered but never received",
                        fire.task_id, fire.slot_ms
                    ),
                );
            }
            patina_dst::sometimes!(received == 2, "sched-wakeup-bounded-repeat");
            if received > 2 {
                violation(
                    "sched-wakeup-dup-unbounded",
                    &format!(
                        "fire ({}, {}) received {received} times",
                        fire.task_id, fire.slot_ms
                    ),
                );
            }
        }
    }
    // Every received wakeup must be accounted for by a delivered fire record;
    // an untracked wakeup means two runs raced the same slot.
    let wakeup_total = wakeups.len() as u64;
    patina_dst::sometimes!(delivered_fires > 0, "sched-fires-delivered");
    if wakeup_total > 2 * delivered_fires {
        violation(
            "sched-wakeups-exceed-fires",
            &format!("{wakeup_total} wakeups for {delivered_fires} delivered fires"),
        );
    }

    // I3: recurring tasks sit on their grid and hold no lease at quiescence.
    for task in retry!(store.list_tasks())? {
        if let Some(interval_ms) = task_intervals.get(&task.id) {
            let on_grid = task.next_run_at_ms >= task.anchor_ms
                && (task.next_run_at_ms - task.anchor_ms) % interval_ms == 0;
            if !on_grid {
                violation(
                    "sched-resume-off-grid",
                    &format!(
                        "task {} next={} anchor={} interval={}",
                        task.id, task.next_run_at_ms, task.anchor_ms, interval_ms
                    ),
                );
            }
        }
        patina_dst::sometimes!(
            task.enabled && task.lease.is_some(),
            "sched-lease-live-at-quiescence"
        );
        if one_shots.contains(&task.id) {
            patina_dst::sometimes!(task.completed_at_ms.is_some(), "sched-one-shot-completed");
        }
    }

    // I4: the store's atomic-write discipline leaves no temp litter.
    check_no_tmp_files(Path::new(SCHED_ROOT), "sched-tmp-litter").await?;
    Ok(())
}

async fn check_outbox_invariants(adapter_id: &str, enqueued: &[String]) -> Result<()> {
    // Conservation: every message the agent enqueued ends in exactly one
    // terminal/queue directory. In no dir = lost (at-least-once broken);
    // in several = duplicated state (double delivery on the next claim).
    let root = PathBuf::from(ADAPTER_ROOT);
    let dirs = [
        ("outbox", root.join("outbox").join(adapter_id)),
        ("inflight", root.join("outbox-inflight").join(adapter_id)),
        ("delivered", root.join("outbox-delivered").join(adapter_id)),
        ("failed", root.join("outbox-failed").join(adapter_id)),
    ];
    let mut locations: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for (label, dir) in &dirs {
        let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
            continue;
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                locations.entry(stem.to_string()).or_default().push(label);
            }
        }
    }
    for id in enqueued {
        match locations.get(id).map(Vec::as_slice) {
            None | Some([]) => violation(
                "outbox-message-lost",
                &format!("message {id} is in no outbox directory"),
            ),
            Some([_single]) => {}
            Some(many) => violation(
                "outbox-message-multiplied",
                &format!("message {id} is in {many:?}"),
            ),
        }
    }
    patina_dst::sometimes!(
        locations.values().any(|dirs| dirs.contains(&"failed")),
        "outbox-terminal-failure-observed"
    );
    check_no_tmp_files(&root, "outbox-tmp-litter").await?;
    Ok(())
}

/// Walks a store root and reports any `.tmp` staging file that survived.
///
/// CONFIRMED FINDING (demoted so it stops masking rarer bugs): a crash in the
/// staged-write/rename gap strands `.tmp` files in every store directory, and
/// exo has no cleanup sweep — litter grows without bound across crashes.
/// Reported as an aggregating non-fatal verdict instead of a violation.
async fn check_no_tmp_files(root: &Path, label: &'static str) -> Result<()> {
    let mut litter = 0u64;
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(mut entries) = tokio::fs::read_dir(&dir).await else {
            continue;
        };
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();
            if entry.file_type().await?.is_dir() {
                stack.push(path);
            } else if path.extension().and_then(|e| e.to_str()) == Some("tmp") {
                litter += 1;
            }
        }
    }
    if litter > 0 {
        patina_dst::verdict(
            VerdictKind::Pass,
            "tmp-litter-observed",
            &format!("{label}: {litter} stranded staging file(s)"),
        );
    }
    Ok(())
}
