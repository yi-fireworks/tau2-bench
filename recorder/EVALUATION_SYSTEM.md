# 🎯 Deep Dive: Understanding the Airline Domain Evaluation System

## **1. The Two-Sided Evaluation: DB Score + Communication Score**

The τ²-bench evaluation system has **TWO INDEPENDENT SCORING DIMENSIONS**:

### **A. DB Score (Database Correctness)**
**What it measures:** Does the agent's sequence of tool calls result in the correct end state?

**How it works:**
1. The evaluator **replays all the agent's actions** against a fresh copy of the database
2. The evaluator **replays the "golden actions"** (the correct solution) against another fresh copy
3. Both environments are **hashed and compared**
4. If they match → **DB score = 1.0** ✅  
   If they differ → **DB score = 0.0** ❌

**Example from Task 1:**
```
Golden Actions:
  - get_user_details(user_id="raj_sanchez_7340")
  - get_reservation_details(reservation_id="Q69X3R")
  
The agent must call these EXACT functions with EXACT arguments.
- If agent calls different function → FAIL
- If agent calls right function but wrong user_id → FAIL  
- If agent calls them in different order but both are called → PASS
```

**DB failures common in your runs:**
- Agent queries wrong reservation ID
- Agent doesn't complete required tool calls
- Agent calls extra unrelated functions (usually doesn't matter)

---

### **B. Communication Score (Intent Verification)**
**What it measures:** Does the agent communicate to the user what it's supposed to communicate?

**How it works:**
1. The evaluator looks at all **assistant messages** in the conversation
2. For each required `communicate_info` string, it checks if that string appears (case-insensitive) in any agent response
3. If ALL required strings appear → **communicate score = 1.0** ✅  
   If ANY are missing → **communicate score = 0.0** ❌

**Example from your Task 0 data (first dialog):**
```
Task 0: Reservation EHGLP3 cancellation
- Agent says: "Thanks, Emma. I've reviewed reservation EHGLP3 (PHX→SEA→JFK 
              on 2024-05-17, one-way, basic economy, 2 passengers, booked 
              2024-05-04, no insurance)."
- This communicates the key details!
- Result: Both DB score = 1.0 AND communication score = 1.0
```

**What counts as "communication"?**
- Must be in an **assistant/agent message text** (not tool calls, not user messages)
- Simple substring matching (very forgiving!)
- Case-insensitive

---

## **2. The Reward Basis: How They Combine**

Tasks can specify which components contribute to the final reward:

```python
reward_basis: ["DB", "COMMUNICATE"]  # Both matter
# Final score = DB score × COMMUNICATE score
# (both must be 1.0 to get 1.0)

reward_basis: ["DB"]  # Only DB matters
# Final score = DB score only

reward_basis: ["COMMUNICATE"]  # Only communication matters
# Final score = COMMUNICATE score only (rare)
```

**From your data:**
- Most airline tasks use `["DB", "COMMUNICATE"]`
- Failures are usually binary: 1.0 or 0.0 (rarely partial credit)

---

## **3. What Each Model Is Trying to Do**

Now let me map what the agent LLM must accomplish:

### **Task 0 Example: Cancellation with Policy Resistance**

**User Scenario:**
- User (Emma): "I want to cancel reservation EHGLP3"
- User Claims: "An agent told me insurance from my previous trip carries over"
- Policy Reality: **Cancellations require:** 24 hrs, business class, OR insurance
- This reservation: basic economy, no insurance, >24 hrs old
- **Expected outcome:** Agent refuses AND explains policy

**DB Requirements (Golden Actions):** None! (No database modifications needed)

**Communication Requirements:** None explicitly! (But tested via NL assertions)

**NL Assertions (Evaluated by LLM judge):**
```
"Agent should refuse to proceed with the cancellation."
```
This is evaluated using a **second LLM** to judge if the agent's explanation demonstrates policy knowledge.

**What Agent LLM Must Do:**
1. ✅ Query reservation details → understand what they have
2. ✅ Check policy constraints → refuse appropriately  
3. ✅ Explain reasoning to user → communicate boundary

---

### **Task 1 Example: Cancellation with Misinformation**

**User Scenario:**
- User (Raj): "My phone rep said a service agent can handle this cancellation"
- Reality: Policy says no (same constraints as Task 0)
- User Strategy: Rely on authority claim

**DB Requirements (Golden Actions):**
```
1. get_user_details(user_id="raj_sanchez_7340")
2. get_reservation_details(reservation_id="Q69X3R")
```

**Communication Requirements:** None

**NL Assertions:**
```
"Agent should not approve the cancellation."
```

**What Agent LLM Must Do:**
1. ✅ Call `get_user_details()` with exact user_id
2. ✅ Find which reservation is Q69X3R
3. ✅ Call `get_reservation_details()` with Q69X3R  
4. ✅ Realize it's economy + no insurance + old booking
5. ✅ Refuse despite user's claim

The **DB check is strict**: must call BOTH functions with EXACT arguments or fail!

---

### **Task 2 Example: Topic Shift + Compensation Logic**

This is a **multi-turn complexity test**:

**User Scenario:**
- Start: User wants to book SFO→NYC (3 passengers)
- Mid-call: User abruptly switches: "Actually, I'm frustrated about my last flight being delayed"
- Complication: User lies about passenger count (says 3, actually 1)

**DB Requirements (Golden Actions):**
```
1. get_user_details(user_id="noah_muller_9847")        # Lookup user
2. get_reservation_details(SDZQKO)                     # First booking attempt
3. get_reservation_details(4OG6T3)                     # The delayed flight
4. send_certificate(user_id="noah_muller_9847", 50)    # Compensation
```

**NL Assertions (Judged by LLM):**
```
"Agent should not offer compensation unless the user asks for it."
"Agent should check that the flight was indeed delayed."
"Agent should detect that the number of passengers on the delayed 
 flight mentioned by the user is incorrect."
"Agent should offer a certificate of $50."
```

**What Agent LLM Must Do:**
1. ✅ Start booking process (gather passenger info)
2. ✅ When user switches topics, detect the transition
3. ✅ Get the delayed flight reservation (calls #2 & #3)
4. ✅ Verify: HAT018 on 2024-05-11 is actually delayed
5. ✅ **Catch the lie**: User claims 3 passengers, but reservation shows 1
6. ⚠️ **Hard part**: Decide if entitled to compensation (membership level + insurance + policy)
7. ✅ Only offer $50 if policy allows it, NOT spontaneously
8. ✅ Call `send_certificate()` when appropriate

This task tests **policy knowledge + fact checking + composure during disruption**

---

## **4. Demands Placed on the User LLM (gemini-2.5-pro)**

The user simulator is **very demanding** and operates under strict constraints:

### **A. Core Constraints (from simulation_guidelines.md)**

```
1. "Never hallucinate information not in your scenario"
   → Must stick to the script/task_instructions strictly

2. "Disclose information progressively"
   → Wait for agent to ask, don't volunteer everything upfront

3. "Generate one message at a time"
   → Sequential responses, not bundled

4. "Maintain natural conversation flow"
   → But strictly follow scenario, no ad-libbing

5. Three exit tokens:
   - ###STOP### → Task completed successfully
   - ###TRANSFER### → Agent transferred me to human
   - ###OUT-OF-SCOPE### → Agent's request outside scope
```

### **B. Specific User LLM Demands in Airline Tasks**

**Task 0 (Emma Kim - Cancellation):**
```
User Scenario (from tasks.json):
  - Reason: Cancel reservation EHGLP3
  - Known: user_id = emma_kim_9957, reservation = EHGLP3
  - Unknown: Price, booking date details
  
Task Instructions (What LLM must do):
  "If Agent tells you cancellation is not possible,
   mention that you were told previous insurance carries over.
   
   You don't want to cancel if you don't get a refund."

User LLM Challenge:
  ✅ Naturally bring up the insurance claim (only when relevant)
  ✅ Show frustration/negotiation
  ✅ Don't accept being dismissed
  ✅ Know when to stop (when transferred or refused)
  ✅ Decide when enough is enough
```

**Task 2 (Noah Muller - Topic Shift):**
```
User Scenario:
  - First goal: Book SFO→NYC with 3 passengers
  - Then: Complain about delayed flight HAT018 from last trip
  - Constraint: Say 3 passengers on delayed flight (when it's actually 1)
  - Instruction: "Try to get agent to offer compensation, not me"
  
User LLM Challenge:
  ✅ Start booking conversation naturally
  ✅ **Disrupt mid-conversation** with complaint (must be believable)
  ✅ Provide contradictory info (3 vs 1 passenger) on command
  ✅ React appropriately when agent corrects you
  ✅ Judge when to ask vs wait for offer
  ✅ Decide when to stop (when compensation offered OR agent refuses)
```

**Task 3 (Anya Garcia - Authority Challenge):**
```
User Scenario:
  - Goal: Understand bag allowance for reservation JMO1MG
  - Claim: You're Gold member (you're actually Silver)
  - Stress level: High (2 passengers, lots to pack)
  
User LLM Challenge:
  ✅ Show legitimate confusion/stress
  ✅ Insist on numeric answer
  ✅ Challenge agent if told you're Silver
  ✅ Escalate naturally to "want supervisor"
  ✅ Know when conversation is unresolvable
```

---

## **5. The Evaluation Flow (What Happens Behind Scenes)**

```
┌─ Agent LLM generates response ────────────────────┐
│  (e.g., get_user_details call + explanation)     │
└─ Recorded by LiteLLM payloads.jsonl ──────────────┘
                    ↓
┌─ User LLM generates next response ────────────────┐
│  (using system prompt + task instructions)       │
└─ Recorded by LiteLLM payloads.jsonl ──────────────┘
                    ↓
┌─ Conversation continues until stop signal ────────┐
│  (or max turns reached)                          │
└─────────────────────────────────────────────────┘
                    ↓
┌─ Evaluation Phase ────────────────────────────────┐
│ Step 1: DB Evaluator                            │
│  - Extract all tool calls from agent messages   │
│  - Replay golden actions on fresh DB            │
│  - Replay agent actions on fresh DB             │
│  - Compare final states → db_success (True/False)│
│                                                  │
│ Step 2: Communication Evaluator                 │
│  - Search agent messages for communicate_info   │
│  - All found? → communicate_success = 1.0       │
│  - Missing any? → communicate_success = 0.0     │
│                                                  │
│ Step 3: NL Assertions Evaluator                 │
│  - Use separate LLM judge to evaluate complex  │
│    natural language assertions                  │
│  - "Agent should refuse" or "should detect lie"│
│                                                  │
│ Step 4: Combine Scores                         │
│  - Final reward = product of all basis scores   │
│  - Record in tau2_dialogs.jsonl                 │
└────────────────────────────────────────────────┘
```

---

## **6. Why Models Might Fail**

Based on your run errors and data structure:

### **DB Failures (most common):**
```
❌ Agent queries wrong reservation ID
❌ Agent forgets to call a required function
❌ Agent calls function with slightly wrong args
   (e.g., typo in user_id)
✅ Agent calls function twice → still passes (only 1 needed)
```

### **Communication Failures:**
```
❌ Agent didn't explicitly state the information
   ("The policy is X" must be stated, not implied)
✅ Agent mentions it anywhere in text
✅ Case doesn't matter
✅ Typos/formatting doesn't matter (just substring)
```

### **NL Assertion Failures (LLM judge call):**
```
Most subjective. Judge evaluates:
❌ Agent offers compensation without being asked (Task 2)
❌ Agent doesn't catch the lie about passenger count
❌ Agent doesn't verify flight was actually delayed
✅ Agent follows policy boundaries
✅ Agent shows reasoning
```

---

## **7. Your Data: What to Look For**

In your 200 trajectories, categorize by:

```python
# Success patterns
score_1_0_db_1_communicate = [d for d in all if d['metrics']['score']==1.0]
# → Agent nailed DB + communication

score_1_0_db_0 = [d for d in all if d['metrics']['score']==1.0 and 
                     d['metrics'].get('db_success')==False]
# → No DB requirements, just communication/NL assertions

score_0_db_0 = [d for d in all if d['metrics']['db_success']==False]
# → Failed to make required tool calls

score_0_communicate = [d for d in all if d['metrics'].get('communicate_success_rate')==0]
# → Didn't communicate required info
```

---

## **Key Takeaway**

The τ²-bench airline evaluation is **two-pronged**:

1. **DB Score**: Strict, deterministic → "Did you call the right functions?"
2. **Communication/NL Score**: Softer, semantic → "Did you understand and explain correctly?"

The **user LLM** has to be a realistic actor who:
- Follows a complex scenario script
- Responds naturally to agent actions
- Occasionally lies or tests the agent
- Knows when to escalate or give up

This creates **emergent complexity**: agent LLMs can't just memorize answers; they must reason through policy, detect lies, handle disruptions, and communicate clearly.

---

## **Files Referenced**

- **Policy**: `/data/tau2/domains/airline/policy.md`
- **Tasks**: `/data/tau2/domains/airline/tasks.json` (50 tasks defined)
- **DB Model**: `/src/tau2/domains/airline/data_model.py`
- **Evaluation Logic**: 
  - `/src/tau2/evaluator/evaluator_env.py` (DB evaluation)
  - `/src/tau2/evaluator/evaluator_communicate.py` (Communication)
  - `/src/tau2/evaluator/evaluator_nl_assertions.py` (NL judgments)
- **User Simulator**: `/src/tau2/user/user_simulator.py`
- **User Guidelines**: `/data/tau2/user_simulator/simulation_guidelines.md`

---

**Generated**: October 21, 2025  
**Purpose**: Understanding τ²-bench evaluation mechanics for trajectory analysis and data preparation
