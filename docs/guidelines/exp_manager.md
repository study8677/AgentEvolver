## 🚀 Overview
Exploration in complex environments is a key challenge for autonomous agents. Traditional reinforcement learning approaches rely heavily on trial-and-error, often generating redundant trajectories and converging slowly. In contrast, humans efficiently leverage past experiences to guide future actions, learning faster and more systematically.

The **`Self-Navigating`** framework adopts this principle by enabling agents to internalize and reuse prior experiences. It shifts exploration from unguided trial-and-error to structured, knowledge-driven self-improvement, improving both learning efficiency and policy quality.

At the core of the framework, the **Experience Manager** oversees all aspects of experience handling, including:

1. **Experience Pool Management** – Constructing and updating the experience pool with new trajectories and summaries.
2. **Experience Mode Control** – Determining whether to add experiences during rollouts and whether to remove experience information during training.
3. **Rollout & Training Context Management** – Providing relevant historical context during rollouts and maintaining experience-strippped training messages.
4. **Training Loss Management** – Aggregating and processing losses with respect to experience-based adjustments for stable learning.

## 🧩 Core Features
The Self-Navigating framework enhances reinforcement learning by transforming how agents **create, reuse, and refine experiences**.
It introduces structured mechanisms that make exploration **more efficient**, **context-aware**, and **self-evolving**.

At its core are two classes:

- **`ExperienceManager`** — handles experience scheduling, allocation, and pool updates.  
- **`ExperienceWorker`** — manages context injection during rollout and cleanup during training.


### 1. Dynamic Experience Allocation

**Purpose**: 
Dynamically decide **how much and when** to use experience during both **training** and **rollout** stages.

**How it works**: 
This module performs two levels of adaptive allocation:

- **Task-Level Allocation**  
    - Determines whether each training task should **keep** or **discard** experience.
    - Controlled by `train_sample_expmode`:  
        - `"allkeep"` → all tasks retain experience  
        - `"alldiscard"` → all tasks discard experience  
        - `"hybrid"` → keep ratio controlled by `train_sample_keepratio`
    - Key Codes:

```python
# Class: ExperienceManager
# Function: allocate_train_mode()
expmode_to_ratio = {
    "allkeep": 1.0,
    "alldiscard": 0.0,
    "hybrid": self.train_sample_keepratio
}
keep_ratio = expmode_to_ratio.get(self.train_sample_expmode, self.train_sample_keepratio)
exp_modes = ['keep'] * keep_count + ['discard'] * (len(tasks) - keep_count)
```

- **Rollout-Level Allocation**
    - Determines the proportion of rollouts within one task that will **include experience**.
    - Controlled by `val_rollout_expmode` and `train_rollout_expmode`:
        - `"woexp"` → no rollout uses experience (pure exploration) 
        - `"all"` → all rollouts use experience (fully guided) 
        - `"mixed"` → *partially guided*, rollout experience usage ratio determined by `rollout_expratio`
    - The parameter `rollout_expratio` only takes effect when `val_rollout_expmode/train_rollout_expmode="mixed"`. For example, `rollout_expratio=0.3` means *30%* of rollouts will include experience, while the remaining *70%* proceed without it.

```python
# Class: ExperienceManager
# Function: allocate_add_exp()
add_exp_choices = {
    "woexp": [False] * rollout_n,
    "mixed": sorted(
        [i < round(rollout_n * self.rollout_expratio) for i in range(rollout_n)],
        key=lambda _: random.random()
    ),
    "all": [True] * rollout_n
}[exp_mode]
```

**✅Effect**:

- `train_sample_expmode` controls how experience is used in training samples.

-  `val_rollout_expmode/train_rollout_expmode` defines the exploration regime (`woexp` / `mixed` / `all`).

- `rollout_expratio` refines `mixed` mode by determining how many rollouts reuse experience.

Together, they enable dynamic balancing between exploration and exploitation.

### 2. Asynchronous Experience Summarization

**Purpose**: 
Convert raw trajectories into summarized experiences **asynchronously**, ensuring continuous learning without blocking.

**How it works**: 

- Periodically triggered by training steps (`updated_freq`).

- Executes summarization jobs via a background thread (`ThreadPoolExecutor`).

- Stores summarized results in the shared experience pool.

```python
# Class: ExperienceManager
# Function: update_experience_pool()
summary_task = self.thread_pool.submit(
    self.em_client.call_summarizer, trajectories=trajectories
)
```


**✅Effect**:
The experience pool grows in parallel with training, keeping the agent’s knowledge base continuously updated.

### 3. Context-Aware Rollout Management

**Purpose**: 
Make rollouts context-aware by injecting relevant past experiences into prompts.

**How it works**: 

- Retrieves top-K related experiences via `EMClient`.

- Formats and prepends them to the rollout message.

- Enhances rollout context without modifying the underlying task.

```python
# Class: ExperienceWorker
# Function: manage_rollout_context()
history_exp = self.em_client.call_context_generator(trajectory)
formatted_exp = self.experience_template.format(history_exp)
trajectory.steps[-1]["content"] = formatted_exp + trajectory.steps[-1]["content"]
```


**✅Effect**:
Each rollout benefits from relevant prior knowledge, reducing redundant exploration.


### 4. Training Context Management

**Purpose**: 
Ensure training messages remain clean by removing injected experience when not needed.

**How it works**: 

- Detects experience templates in training messages using regex.

- Removes them when `train_mode="discard"` while retaining extracted text for analysis.


```python
# Class: ExperienceWorker
# Function: manage_training_context()
pattern = re.escape(self.experience_template).replace(r'\{\}', '(.*?)')
cleaned_message = re.sub(pattern, '', message, flags=re.DOTALL)
```


**✅Effect**:
Guarantees that training data integrity aligns with the current experience policy.




### 5. Training Loss Processing

**Purpose**:  
Ensure stable policy updates when mixing **on-policy rollouts** and **off-policy experience replays**, allowing the agent to leverage past trajectories without destabilizing learning.

**How it Works**:

- This module computes a *heterogeneous PPO loss* that combines:
    - **On-policy loss**: derived from fresh rollouts.  
    - **Off-policy loss**: derived from experience-augmented samples.  
- An **experience mask (`exp_mask`)** distinguishes the two. Each loss is clipped and optionally adjusted for negative advantages. Finally, the losses are combined and aggregated according to `loss_agg_mode`.

```python
# Function: het_compute_token_on_off_policy_loss()

# 1️⃣ Compute policy ratio and approximate KL divergence
negative_approx_kl = log_prob - old_log_prob  # difference between new and old log-probabilities
ratio = torch.exp(negative_approx_kl)       # policy ratio r_t = exp(log_pi_new - log_pi_old)
ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)  # approximate KL divergence

# 2️⃣ Compute on-policy losses (exp_mask = 0)
on_pg_losses, _, _ = compute_pg_losses(cliprange_low, cliprange_high)
# Mask out experience tokens to ensure only fresh rollouts contribute
on_pg_loss = verl_F.masked_mean(on_pg_losses, (1.0 - exp_mask) * response_mask)

# 3️⃣ Compute off-policy losses (exp_mask = 1)
off_pg_losses, _, _ = compute_pg_losses(off_cliprange_low, off_cliprange_high)
# Mask to include only experience tokens
off_pg_loss = verl_F.masked_mean(off_pg_losses, exp_mask * response_mask)
# Ensure numerical stability
off_pg_loss = torch.tensor(0.0) if off_pg_loss.isnan().item() else off_pg_loss

# 4️⃣ Combine both losses using the experience mask
pg_losses = off_pg_losses * exp_mask + on_pg_losses * (1.0 - exp_mask)
# Aggregate token-level losses according to selected mode (e.g., "token-mean")
pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
```

**✅Effect**:

- `exp_mask` separates on-policy and off-policy contributions cleanly.

- `off_cliprange_high` enforce trust regions for stable updates.


## ⚙️ Key Parameters & Configuration

### **1. Experience Maker (`experience_maker` submodule)**

**`base_url`** (*str*)  
: The endpoint of the **Experience Maker service**, responsible for summarization and retrieval.  
Example: `"http://127.0.0.1:8001"`.

**`workspace_id`** (*str*)  
: Identifier for the current workspace. Each workspace maintains an isolated experience pool.  
Default: `"default"`.

**`enable_summarizer`** (*bool*)  
: Whether to enable **asynchronous summarization** of raw trajectories into experience snippets.  
`True` enables automatic knowledge distillation from rollouts.

**`enable_context_generator`** (*bool*)  
: Whether to enable **experience retrieval and context injection** during rollouts.  
When `True`, the system fetches top-k relevant experiences and prepends them to rollout prompts.

**`retrieve_top_k`** (*int*)  
: The number of top relevant experiences retrieved per rollout when `enable_context_generator=True`.  
Default: `3`.

**`updated_freq`** (*int*)  
: Frequency (in training steps) for updating the experience pool.  
`0` disables periodic updates.

**`val_summarizer_save`** (*bool*)  
: Whether to save summarized experiences during validation.  
Recommended to be `True` when analyzing model generalization or debugging experience evolution.

---

### **2. Experience Manager (`exp_manager` submodule)**

**`val_rollout_expmode`** (*str*)  
: Controls experience usage in **validation rollouts**.  
Options:  
- `"woexp"` → no experience (pure evaluation)  
- `"mixed"` → partial experience injection (ratio defined by `rollout_expratio`)  
- `"all"` → all rollouts use experience  

**`train_rollout_expmode`** (*str*)  
: Same as `val_rollout_expmode` but applies to **training rollouts**.  
This parameter switches between unguided exploration (`woexp`) and experience-guided rollout (`mixed` or `all`).

**`rollout_expratio`** (*float*)  
: Ratio of rollouts that include experience when `train_rollout_expmode` or `val_rollout_expmode` is set to `"mixed"`.  
Example: `0.3` means 30% of rollouts reuse experience; 70% remain exploratory.  
Default: `0.0`.

**`train_sample_expmode`** (*str*)  
: Defines how **training samples** handle experience after rollout.  
Options:  
- `"allkeep"` → retain all experience information  
- `"alldiscard"` → strip all experience context  
- `"hybrid"` → partial retention (ratio by `train_sample_keepratio`)  

**`train_sample_keepratio`** (*float*)  
: When `train_sample_expmode="hybrid"`, controls the proportion of training samples that keep experience.  
Default: `1.0`.

**`experience_template`** (*str*)  
: Template used to insert retrieved experiences into rollout messages.  
The `{}` placeholder is replaced by formatted experience text.  
Example:  `"\n\nSome Related Experience to help you to complete the task:<EXP>{}</EXP>\n\n"`

**`init_experience_before_training`** (*bool*)  
: Whether to **initialize the experience pool** before training starts.  
Useful when preloading prior knowledge for warm-start training.

**`init_experience_only`** (*bool*)  
: If `True`, the system only initializes the experience pool without starting training.  
Ideal for precomputing embeddings or testing summarization quality.


## 🧭 Quick Start & Recommended Configuration

### **Step 1: Set ReMe Service**
[TODO]

### **Step 2: Recommended Configuration**

```yaml
experience_maker:
  base_url: "http://127.0.0.1:8001"         # Experience Maker service endpoint
  workspace_id: "default"                   # Unique workspace identifier
  enable_summarizer: False                  # Enable async summarization of trajectories
  enable_context_generator: False           # Enable retrieval & context injection during rollout
  retrieve_top_k: 3                         # Number of top relevant experiences retrieved per rollout
  updated_freq: 0                           # Update frequency of experience pool (every k steps; 0 = disabled)
  val_summarizer_save: False                # Save summarized experiences during validation

exp_manager:
  val_rollout_expmode: "woexp"              # Validation rollout mode: ["woexp"=no exp, "mixed"=partial, "all"=full]
  train_rollout_expmode: "mixed"            # Training rollout mode: ["woexp", "mixed", "all"]
  rollout_expratio: 0.5                     # Ratio of rollouts using experience in "mixed" mode (e.g., 0.5 = 50%)
  train_sample_expmode: "alldiscard"        # How to handle experience in training samples: ["allkeep", "alldiscard", "hybrid"]
  train_sample_keepratio: 0.0               # Keep ratio for "hybrid" mode (0.0 = discard all)
  experience_template: "\n\nSome Related Experience to help you to complete the task:<EXP>{}</EXP>\n\n"  # Experience insertion format
  init_experience_before_training: True     # Initialize experience pool before training starts
  init_experience_only: False               # Only initialize experience pool (no training)

```


  [TODO: more details about experience pool initialization & experience pool example]