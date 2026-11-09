# Prior-art audit: "the swarm that dreams and doubts" (round-3 concept)

> Repo copy, added 2026-11-09, text unchanged from 2026-11-08. It is a snippet-level audit: the only claim it supports is "no direct match found in the sources below", never "first".

Written 2026-11-08. Status: **DRAFT, snippet-level.** Facts are marked `[V]` (the page itself was opened and its title or abstract read in this session) or `[S]` (only a web-search snippet was seen; not yet read in full). Nothing here is a claim of "first". The only claim we may make, if the audit holds, is "no direct match found in the sources below".

## 1. The concept in one paragraph

A defender swarm for instant payouts that (a) builds a **world model** of the platform and its attackers from its own event log, (b) **dreams**: samples attackers (including untested regions) from that model and evolves defenders against them, (c) has a **Skeptic** role that estimates where the world model is unreliable, and (d) **calibrates trust in its own imagination**: after each real round it compares predicted with realized performance, and the promotion margin for a dreamed improvement grows with the gap. Reality = the simulator's ground truth with delayed labels; the PayPal sandbox rail provides real payout receipts and webhooks for the live demo, not the labels.

## 2. What already exists (ingredient by ingredient)

| Ingredient | Closest work found | Status | What it does NOT have (the delta) |
|---|---|---|---|
| LLM self-reflective agent improving a fraud detector | **SAGE**, arXiv 2606.08146, "An LLM-driven Self Reflective Agentic Framework for Fraud Detection" https://arxiv.org/abs/2606.08146 | `[V]` title and abstract opened | Abstract is about automated model/feature search with semantic awareness; no adversary population, world model, swarm roles or payout rail seen in the abstract. Full text not read. |
| LLM-guided rule generation for fraud | GPT-Guided MCTS for symbolic regression in fraud detection, arXiv 2411.04459 https://arxiv.org/html/2411.04459v1 | `[S]` | Static data, no adaptive attacker. |
| Genetic algorithms evolving fraud rules | USPTO 7657497, 8321341, 7840520 ("Online fraud prevention using genetic algorithm solution") https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7657497 | `[S]` | Pre-LLM; no swarm, no world model. |
| Simulated attacks used to retrain a fraud model | USPTO 12597034 "Fraud detection methods and systems based on evolution-based black-box attack models" https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12597034 | `[S]` | Closest to "defender improves against simulated attackers"; no world model, no trust calibration. |
| RL fraudster used as stress tester | El-Awady, adaptive stress testing, arXiv 2107.03577 https://arxiv.org/abs/2107.03577v1 | `[S]` | Attacker only; no co-evolving defender swarm. |
| Attacker/detector multi-agent synthetic fraud | "Catch me if you can: a multi-agent synthetic fraud detection framework" https://research.buaa.edu.cn/en/publications/catch-me-if-you-can-a-multi-agent-synthetic-fraud-detection-frame/ | `[S]` | Closest overall to our current (round-1/2) design. Code availability unknown. |
| Adversarial fine-tuning of fraud detectors | ProFraudGuard (Amazon Science) https://cdn.amazon.science/ce/60/7b7d696149fd85ed3fc105e30364/scipub-approval152129-48384376-profraudguard-proactive-adversarial-finetuning-of-fraud-detectors-with-inverse-reinforcement-learning.pdf | `[S]` | Alternating generator/detector training; no world model of the platform, no calibrated trust. |
| Self-evolving agent workflows / roles / topology | MetaGen arXiv 2601.19290 `[V]` https://arxiv.org/abs/2601.19290 ; Mimosa arXiv 2603.28986 `[V]` https://arxiv.org/abs/2603.28986 ; MOSS (source-level self-rewriting) arXiv 2605.22794 `[V]` https://arxiv.org/abs/2605.22794 ; SEW arXiv 2505.18646 `[S]` ; EvoAgentX `[S]` | titles `[V]`, content `[S]` | None is a fraud defense, none has an adversary population or a payout rail. **Stage-6 "self-evolving" as a mechanism is crowded; we must not present it as the novelty.** |
| Agents trusting / calibrating a world model | Dual-Frontier, arXiv 2609.26293 `[V]` title https://arxiv.org/abs/2609.26293 ; HALT slides (NeurIPS 2026) `[S]` https://neurips.cc/media/neurips-2026/Slides/166141.pdf ; geodesic trust signal (NeurIPS 2026) `[S]` | mixed | General RL and robotics settings; no security/fraud, no multi-agent swarm roles. |
| Quality-diversity co-evolution (alternative idea, not chosen) | Adversarial MAP-Elites https://arxiv.org/html/2505.06617v1 `[S]` ; metamorphic malware with MAP-Elites https://www.napier.ac.uk/research-and-innovation/research-search/outputs/automatic-generation-of-adversarial-metamorphic-malware-using-map-elites `[S]` | `[S]` | Not fraud. |
| Payment simulators | PaySim https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2640900 `[S]` ; MultiMAuS https://github.com/lmzintgraf/MultiMAuS `[S]` | `[S]` | Simulators only. |

## 3. The claim we may be able to make (and its limits)

- **Candidate claim:** "We found no direct match for a defender swarm that builds a world model of payout fraud, dreams attackers from it, and calibrates its trust in those dreams against real outcomes." Every ingredient exists elsewhere; the combination in payouts fraud was not found.
- **Not claimable:** "first", "novel", "nobody has built". Stage-6 self-evolution (workflows, roles, topology, source rewriting) is well published.
- **Search limits:** the search tool is US-only and returned thin results; absence is not proof.

## 4. Corrections to what I told the user in chat

- I linked "ST-EVO" to https://www.preprints.org/manuscript/202603.0129/v1. **That page is "A Self-Reflective Multi-Agent Collaboration Framework for Dynamic Software Engineering Tasks", not ST-EVO.** The ST-EVO name is dropped until its real URL is verified.
- I said real PayPal outcomes anchor the calibration check. **They do not**: labels come from the simulator's delayed ground truth. The sandbox rail proves the swarm's actions on real payouts (5 receipts, 15 verified webhooks), nothing more.

## 5. Open audit items (not done yet)

1. Read the full text of SAGE (2606.08146), "Catch me if you can", ProFraudGuard and USPTO 12597034; record exactly what each does.
2. Search GitHub (topics: fraud, multi-agent, co-evolution, world model) and the Devpost gallery of this hackathon for similar entries.
3. Search for "world model" + "red teaming" / "intrusion detection" / "anomaly detection" to test the security-side gap, since the one search run covered it only thinly.
4. Verify the HALT and geodesic-trust sources by opening them.
5. Re-check emergent-communication and quality-diversity gaps only if those ideas are revived.
