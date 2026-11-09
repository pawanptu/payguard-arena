/** Glossary: one definition per term, shared by every InfoTip (docs/drafts/ui_reviews/content.md). */
export const TIPS = {
  simulated: "Made-up payouts and attacks from a simulator. This site replays recorded runs and never sends payouts.",
  release: "The payout is paid now.",
  stepUp: "The payout waits for extra verification before it is paid.",
  hold: "The payout is held for a human to review.",
  swarm:
    "Four small agents (velocity, graph, drift, policy) that each score a payout. Their scores are combined into one risk score.",
  anomalyLayer: "A second check that can raise a payout's severity but never lower it.",
  anomalyWhat: "Nine checks, with limits set from honest traffic only.",
  trustWeight: "How much the swarm counts this agent's score; it rises and falls with the agent's track record.",
  shareOfRisk: "How much of the total risk this agent supplied.",
  agentScore: "How strongly this agent flagged the payout, from 0 to 1.",
  leakage: "The share of attack value that was paid out.",
  honestHeld: "The share of honest payouts put on hold.",
  honestHeldKpi: "Legitimate payouts that waited for review. This is friction, not an error: the price of catching fraud.",
  honestNeverPaid: "The share of honest payouts that were never paid.",
  score: "Fraud let through (%) + half of honest payouts held (%) + honest payouts never paid (%). Lower is better.",
  preRegistered: "The pass rule was written down before the results were seen.",
  verdictWords: "NO-GO: the rule was not met. GO: it was met.",
  h1h2:
    "Two tests fixed in advance. H1: beat static rules and the best single agent by 3.0 points. H2: beat the version without a world model by 2.0 points.",
  neverSeen: "An attack type the defenders were never trained or tuned on.",
  staticRules: "A fixed set of hand-written rules, tuned once on training data: the simple baseline.",
  ablation: "The same defender with one part removed, to show what that part adds.",
  worldModel: "The dreamer's learned model of how known attacks behave, used to imagine new attacks to train against.",
  dreamer: "The round 3 defender that also trains against imagined attacks.",
  range95:
    "Where the average difference would land if the 20 episodes were resampled; a range that excludes 0 is unlikely to be chance.",
  pooled: "All episodes added together into one number.",
  spent: "A test set already used, so it cannot be a clean test again.",
  singleAgent: "One of the four agents used alone, with no swarm.",
  postHoc: "Designed after the result on that test was known.",
  grounding: "AI text may only use numbers from this case's evidence. If it uses any other number, a standard sentence is shown instead.",
  template:
    "Written from a fixed template using this case's numbers. Shown when there is no AI text yet, or when the AI text used a number that is not in the evidence.",
  liveSim: "LIVE: a real PayPal sandbox call with test money. SIM: simulated.",
  signature: "PayPal's signature on the webhook was checked before it was recorded. Only ids and statuses are kept.",
  readOnly: "The assistant can look things up. It cannot send or change anything.",
  payguardLookup:
    "The PayPal Agent Toolkit has no Payouts tools, so payout status comes from our own read-only GET request.",
  scoreboard:
    "A call is right when it holds back an attack (Hold or Step-up) or releases an honest payout. Agents are scored on the cases you reviewed. Saved in this browser only.",
  caveats: "K1 is a check on training data that had to pass before the sealed test. Folds are splits of that training data.",
} as const;

/** Risk score tip with the real thresholds from the queue data. */
export const riskTip = (stepUp: string, hold: string) =>
  `A number from 0 to 1. From ${stepUp} a payout needs extra verification (step-up line). From ${hold} it is held (hold line).`;
