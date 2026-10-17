# Four-key agent-assisted PC control (MVP design)

Version 3 (2026-10-17). Versions 1 and 2 of the same day are replaced by the operator decisions in §8. Version 3 swaps the key meanings: **R selects and L undoes**. In manual mode, L also returns to the agent. Status: **proposed. Nothing is implemented or tested.** No result is claimed.

**Code location.** The implementation will live in a new, separate repository. No code for this module is written in this repository. This document stays here until that repository exists.

## 0. Scope

This module lets a user operate a macOS computer with four discrete inputs, assisted by a computer-use agent. It does not depend on the wristband. Its input is four abstract events, and any source can produce them:

| Event | Meaning |
|---|---|
| **U** | previous option |
| **D** | next option |
| **R** | select / confirm |
| **L** | undo / stop |

During development the keys F9–F12 produce the events (F9→U, F10→D, F11→L, F12→R). Later the wristband's direction classifier will produce them (UP→U, DOWN→D, LEFT→L, RIGHT→R).

**The MVP aims to make the module work.** Speed and option-set optimisation are not goals. These are out of scope:
- the four-group language-model letter decoder;
- continuous stroke detection;
- wristband integration;
- Windows and Linux;
- more than one user.

Background for this design:
- Prior-art reports: [`reports/Gesture steered computer agents.md`](../../reports/Gesture%20steered%20computer%20agents.md) and [`reports/Four gesture LLM text entry.md`](../../reports/Four%20gesture%20LLM%20text%20entry.md), with their notes in `research_notes/`.
- The wristband evidence is the exploratory W02 development record ([2026-10-15](../results/2026-10-15-w02-exploratory-development.md)). Its numbers are single-participant, presegmented and tuned on the development wears.

## 1. Full-control statement and its basis

The module has two layers.

- **Manual layer, which guarantees completeness.** macOS Switch Control is built in. It scans interface items, offers point scanning (a vertical then a horizontal line sweep that can place the pointer anywhere), and can act through the on-screen Accessibility Keyboard. Keyboard keys can be assigned as switches. Together these produce any pointer position, click, drag and key press ([Apple: Use Switch Control on Mac](https://support.apple.com/en-in/guide/mac-help/mh43607)).
- **Agent layer, which adds efficiency.** A computer-use agent turns multi-step operations into a few menu choices. The completeness claim does not depend on it.

**Claim to be supported:** "every mouse and keyboard input on macOS is reachable with four inputs (manual layer), and the agent layer completes a defined task suite." The claim is limited by the outcome of the bench checks in §7. Before the user session (the login window, pre-boot disk unlock), Switch Control has not been checked and is excluded.

## 2. Components

| ID | Component | Source | Work |
|---|---|---|---|
| C1 | Input adapter: emits U/D/L/R from F9–F12, from a replay file, or through an error injector (§6) | new | small |
| C2 | Menu overlay: an always-on-top window showing the question, the options, the highlight, the plan preview, the execution status and the confirm windows (§3.3). In manual mode it shows only a note that L returns to the agent | new | small |
| C3 | Agent loop: Claude with the computer-use tool and one custom tool, `choose` (§4) | Claude API + new glue | main work |
| C4 | Screen and input driver for macOS (screenshot, click, type, key press) | existing driver, e.g. the computer-use reference executor or Cua Driver | integration |
| C5 | Undo stack: menu steps and executed actions, each with a reversibility class | new | small |
| C6 | Manual-mode handover: enables Switch Control on entry; on L, disables it and returns to the agent | macOS settings + a small script | depends on bench check B2 |
| C7 | Logger: events, menus, options, selections, actions, screenshots, timestamps | new | small |

The model is Claude, chosen as the most convenient at present. Open-source models and agent frameworks may replace it later. C3 keeps the model behind one interface so it can be swapped.

## 3. Interaction

### 3.1 States

| State | Listens to | Purpose |
|---|---|---|
| MENU (top, compose, plan preview, clarify, confirm, result, after-stop) | U, D, L, R | All choices |
| EXECUTING | only L | Agent acts; L asks to stop it |
| CONFIRM WINDOW | R confirms; any other event cancels | Guards stop and undo against a single stray event |
| MANUAL | only L, which returns to the agent (Switch Control has the other keys) | Direct control |

```
TOP ──R──▶ COMPOSE (repeated menus) ──▶ PLAN PREVIEW ──R──▶ EXECUTING ──▶ RESULT ──▶ TOP
                                            │L                  │ L
                                            ▼                   ▼
                                       back one menu      "Stop the agent? R = confirm"
                                                                │ R                │ other event
                                                                ▼                  ▼
                                                     AFTER-STOP MENU (refreshed)   resume EXECUTING
                                                                │ L
                                                                ▼
                                                     "Undo <action>? R = confirm" ──R──▶ revert, AFTER-STOP MENU again
```

### 3.2 Menus

- The highlight starts on the agent's top-ranked option. **U** and **D** move it to the previous and next option; the list does not wrap.
- **R** selects the highlighted option. **L** undoes the last menu selection, returning to the previous menu.
- **Composing** builds an intent through successive menus, for example "Email" → "Reply to the newest message from Alice" → "Accept the meeting" → "Send".
- **Plan preview** lists the steps the agent will take and marks every irreversible step. R starts execution and L returns to composing.
- During execution the agent opens a **clarify** menu whenever it is unsure. Before each irreversible step it opens a **confirm** menu, where the highlight starts on "Not yet", so a single stray R cannot confirm.

Fixed entries follow the agent's options in every menu:

- **None of these:** the agent must offer a different set of options. It may not ask for free text.
- **Manual control:** switches to manual mode (§3.5).

In the top menu only, a **Pause** entry ignores all input until a deliberate resume sequence, L L L. The resume sequence must be checked against false triggers once the wristband is the source; for the MVP it is a placeholder.

### 3.3 Execution: stop and undo through confirm windows

While the agent executes, U, D and R are ignored.

1. **L** opens a confirm window: "Stop the agent? R = confirm". The agent pauses after its current atomic action (one click or key press) while the window is shown.
2. **R** confirms. The agent stops, and the overlay returns to an **after-stop menu**. Any other event closes the window, and the agent resumes.
3. The agent refreshes the after-stop menu for the new screen state: what to do next, or instead. It offers options such as "Resume the plan" or "Do X instead", followed by the fixed entries, and is navigated like any menu.
4. **L** in the after-stop menu opens an undo confirm window: "Undo <action>? R = confirm". R carries out the revert (§3.4); any other event cancels it. Either way the user returns to a refreshed after-stop menu, so each further L, R pair undoes one more action.

Confirm windows have no time-out. They stay open until the next event.

### 3.4 Undo semantics

| Most recent entry | Undo (L) |
|---|---|
| A menu selection | Returns to the previous menu. No effect on the computer |
| An executed **reversible** action (L in the after-stop menu, then R in the confirm window) | The agent carries out the revert (e.g. Cmd-Z, Back, close a window) |
| An executed **irreversible** action | The confirm window says "Cannot be undone" instead; any event returns to the after-stop menu |

The agent assigns the class (reversible or irreversible) to every step during plan preview. If unsure, it chooses irreversible.

### 3.5 Manual mode (Switch Control)

Switch Control is **enabled only after the user selects Manual control**, and it is disabled again on return. While it is enabled, C1 ignores U, D and R. Switch Control uses manual scanning (auto-scan off) and exactly four switch actions:

| Event | Switch Control action |
|---|---|
| U | Restart scanning (from the top) |
| D | Move to next item |
| R | Select item |
| L | Stop scanning, and return to the agent |

**Return path.** L does both. Switch Control stops scanning, and C1 also receives the L event. C6 then disables Switch Control, and the agent resumes with a refreshed top menu for the current screen state. No fifth action and no on-screen button are needed. Bench check B2 must confirm that C1 still receives the L key while it is assigned as a Switch Control switch.

Text that the agent cannot offer as options, such as passwords or arbitrary content, is typed in manual mode with the Accessibility Keyboard.

## 4. Agent contract

There is one custom tool:

```
choose(kind, question, options[], default_index) -> selected index | "none" | "manual" | "undo"
  kind: top | compose | plan | clarify | confirm | result | after_stop
  options: short labels (≤ 60 characters), ranked, at most 6; the fixed entries are added by C2
```

The rules are enforced in the system prompt and also in code:

1. Every user interaction goes through `choose`. The agent never asks for free text. C3 rejects any other path that asks the user something.
2. Options are ranked by likelihood. They are concrete intents or contents, not open questions.
3. Text content is offered as complete candidate strings, such as replies, search queries or filenames. If none fits, the user takes "None of these" or "Manual control".
4. A plan preview is required before any action on the computer. Steps are labelled reversible or irreversible.
5. The agent asks for confirmation before each irreversible step, and in clarify mode whenever it is unsure.
6. The agent checks for a stop request after every atomic action.
7. No credential entry and no payments by the agent. These happen in manual mode.
8. The MVP runs in a dedicated macOS test user account with no access to personal data.

## 5. Session example (illustrative, not a result)

1. **Top menu:** "Reply to email" (highlighted), "Open a file", "Search the web", "Change a setting". R.
2. **Compose:** "Newest from Alice: 'Friday meeting?'" (highlighted), ... R.
3. **Compose:** "Accept: 'Friday works, see you then.'", "Decline politely", "Propose another time". R.
4. **Plan preview:** 1. open the message (reversible); 2. click Reply (reversible); 3. type the text (reversible); 4. Send (**irreversible**). R.
5. **Confirm before Send:** highlight on "Not yet". D, then R.

That is 6 events. Had the user changed their mind during step 3 of the plan: L opens "Stop the agent?", R confirms, and the refreshed after-stop menu appears. L, then R in the undo confirm window, removes the typed text. The menu then offers, for example, "Decline politely instead".

## 6. Evaluation plan for the MVP

**Task suite:** about 10 fixed everyday tasks, such as opening a file, replying to an email, a web search, changing a system setting and renaming a file. Tasks are written before the first run and are not changed after it.

**Conditions:**

| Condition | Input | Status of results |
|---|---|---|
| K0 | F9–F12, no injected errors | Functional test |
| K1 | F9–F12, with each event replaced according to the wristband's direction confusion matrix | Simulation. Exploratory, because the confusion matrix comes from the W02 development wears |
| W | Wristband | Later. Requires live capture and is not part of the MVP |

The K1 injector draws replacements from the development-wear confusion matrix of the chosen calibrated recipe, with a recorded seed. The sealed wears W07, W10 and W12 are never used for it.

**Recorded per task:**
- completed (yes/no);
- events, menus, "None of these" selections, stops, undos and manual-mode entries;
- wall time;
- unintended actions, split into reversed and not reversed.

A completion counts only when an independent check of the final state passes (file exists, message sent to the test inbox, setting value). The agent's own report does not count.

**Full-control demonstration:** at least one task completed entirely in manual mode, showing that the manual layer alone suffices.

## 7. Bench checks before coding

| ID | Check | Why it matters |
|---|---|---|
| B1 | F9–F12 can be Switch Control switches with the four actions in §3.5. F11 is bound to Show Desktop by default and must be unbound; on laptop keyboards F9–F12 may need the Fn setting | Mapping in manual mode |
| B2 | Switch Control can be enabled when Manual control is selected and disabled by C6 on L. C1 still receives F11 (L) while it is a Switch Control switch; if not, a fallback return path is needed | The two layers must never act on the same event, except L in manual mode |
| B3 | The driver (C4) can act while the overlay (C2) is shown, and C1 can capture F9–F12 globally without passing them to other apps | Agent mode works at all |
| B4 | Claude obeys the `choose`-only rule over a few scripted sessions | Without it the design fails |

Each check result is recorded with the date, macOS version and outcome before implementation starts.

## 8. Decisions

**Made by the operator (2026-10-17):**
- Key meanings: U previous, D next, R select/confirm, L undo/stop.
- Manual mode uses four Switch Control actions: U Restart scanning, D Move to next item, R Select item, L Stop scanning. L also returns to the agent.
- Switch Control is enabled only in manual mode.
- During execution only L is accepted:
  - L opens a stop confirm window, and R confirms.
  - The agent pauses while that window is shown.
  - After a stop, the user returns to a refreshed menu of what to do next or instead.
  - L there opens an undo confirm window, and R confirms.
- Confirm windows have no time-out.
- Claude is the model for now; open-source models and frameworks may follow.
- F9–F12 are the development keys.
- The code goes in a new repository.

**Open:**
- The API cost budget, to be decided later.
