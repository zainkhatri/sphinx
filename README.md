# Morty Rescue Mission

Save 1000 Morties by choosing which planet portal gives the best odds. Each planet's success rate drifts over time, so the trick is balancing exploration, commitment, and group size.

## Core Idea

Thompson Sampling with exponential discounting (gamma) to adapt to changing success rates.
Send more Morties when confident, fewer when uncertain.

## Algorithm Strategy

The core strategy is based on the multi armed bandit problem with some specific adaptations:

1. **Exploration Phase**: Try each planet twice to gather initial data
2. **Locking Phase**: Identify the best performing planet and commit to it for a period
3. **Adaptive Decision Making**: Only switch planets if performance significantly degrades
4. **Group Size Optimization**: Send more Morties at once when confident, fewer when uncertain

Key insights:
- Switching planets too frequently prevents capitalizing on good options
- Staying too long on a declining planet leads to unnecessary losses
- The sweet spot is locking in early to a good planet and only bailing if it's clearly failing

## Version Highlights

### v1–v3 : Naive Thompson
- Pure TS, no safety checks.
- Constantly switched arms; unstable; mid-50% average.

### v4 : Discounted TS
- Added exponential forgetting (gamma=0.995–0.998).
- Adapted better to drift but still over-reacted sometimes.

### v5 : Credible Bounds
- Added lower-bound checks before sizing up.
- Cut losses from bad 3-sends.

### v6 : Dark Age Detection
- If all planets look bad, force size-1 sends.
- Prevented wipeouts during global slumps.

### v7 : Simple TS + Dark Age (best run)
- Cleanest, minimal version; gamma=0.997, probe every 15.
- Sizing via credible bounds: LB_2=0.53, LB_3=0.60.
- Averaged low 70s; peaked at 75.4%.

### Locking Variant (this file)
- Explores each planet twice, then locks onto the best one.
- Unlocks only if recent success < 20% or two fails in a row.
- Safer and more stable but caps out around 65–70%.

### UCB Mix / Conservative Gates
- Tested 25% UCB exploration and stricter sizing (2+ only >70%).
- Both made it safer but slower; rarely hit 70%.

That's it. v7 gave the best balance of confidence and patience; this locked version trades a few points of success for predictability.
