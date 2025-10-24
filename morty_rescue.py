#!/usr/bin/env python3
"""
Morty Rescue - Save 1000 Morties by choosing the best planet portal

THE PROBLEM:
- We have 1000 Morties to rescue
- There are 3 planets we can send them to
- Each planet either saves or loses the Morties we send
- Planet success rates CHANGE over time (what works now might not work later)

OUR STRATEGY:
1. Try each planet twice to see which one is best (exploration)
2. Pick the winner and stick with it for a while (commitment)
3. Only switch if things get really bad (patience)

WHY THIS WORKS:
- Switching planets too much = bad (you never commit to the winner)
- Staying too long on a dying planet = bad (you lose too many Morties)
- The sweet spot: Lock early, stay locked, only bail if it's truly dying
"""

import os
import requests
import time
import json
import random
from typing import Dict
from math import sqrt, log
from dotenv import load_dotenv

# Load settings from .env file
load_dotenv()

# API Configuration - where to send our requests
BASE_URL = os.environ.get("BASE_URL", "https://mortyexpress.sphinxhq.com")
API_TOKEN = os.environ.get("MORTY_TOKEN")
SLOW_MODE = bool(int(os.environ.get("SLOW_MODE", "0")))  # For testing: adds delays

# ============================================================================
# TUNING KNOBS - These numbers control how the algorithm behaves
# ============================================================================

# Memory: How much to remember old data vs focus on recent results
GAMMA = 0.995  # 0.995 = remember 99.5% of old data each step (slow forgetting)

# Starting beliefs: What we think before seeing any data
PRIOR_A = 0.5  # Start with weak beliefs (let real data dominate quickly)
PRIOR_B = 0.5
MIN_ESS = 2.0  # Always act like we have at least 2 data points

# Locking rules: When to commit to a planet (THE KEY TO SUCCESS!)
WINNER_GAP = 0.08      # Lock if best planet is 8% better than worst (aggressive!)
WINNER_MIN_RATE = 0.50 # Lock if best planet has 50%+ success (aggressive!)
WINNER_STICK = 0.98    # 98% chance to lock when conditions are met
LOCK_DURATION = 60     # Stay locked for 60 trips (long commitment!)

# Group sizing: How many Morties to send at once
# Send more when confident, send 1 when uncertain
HOT_THRESHOLD = 0.58        # 58%+ success → consider sending 2
VERY_HOT_THRESHOLD = 0.68   # 68%+ success → consider sending 2
BLAZING_THRESHOLD = 0.78    # 78%+ success → consider sending 3
SUPER_HOT_THRESHOLD = 0.88  # 88%+ success → definitely send 3
COLD_THRESHOLD = 0.30       # Below 30% → only send 1

# ============================================================================
# PLANET INFO - The 3 planets we can choose from
# ============================================================================
PLANETS = ["cob_planet", "cronenberg_world", "purge_planet"]
PLANET_NAMES = {
    "cob_planet": "On a Cob Planet",
    "cronenberg_world": "Cronenberg World", 
    "purge_planet": "The Purge Planet"
}
PLANET_TO_INT = {  # API needs numbers, not names
    "cob_planet": 0,
    "cronenberg_world": 1,
    "purge_planet": 2
}


# ============================================================================
# HELPER FUNCTION - Random sampling for exploration
# ============================================================================
def beta_sample(alpha: float, beta: float) -> float:
    """
    Sample a random number from a Beta distribution.
    
    SIMPLE EXPLANATION:
    - We track "alpha" (successes) and "beta" (failures) for each planet
    - This function picks a random success rate based on what we've seen
    - More data = less randomness, less data = more randomness
    - This randomness helps us explore uncertain options
    """
    # Edge case: if too little data, just return the average
    if alpha < 0.1 or beta < 0.1:
        return alpha / (alpha + beta)
    
    # Math helper function (don't worry about the details)
    def gamma_sample(shape):
        if shape < 1:
            return gamma_sample(shape + 1) * (random.random() ** (1.0 / shape))
        
        d = shape - 1.0/3.0
        c = 1.0 / sqrt(9.0 * d)
        
        while True:
            while True:
                u1, u2 = random.random(), random.random()
                z = sqrt(-2.0 * log(u1)) * (1 if u2 < 0.5 else -1)
                x = 1.0 + c * z
                if x > 0:
                    break
            
            v = x * x * x
            u = random.random()
            
            if u < 1.0 - 0.0331 * (z * z) * (z * z):
                return d * v
            
            if log(u) < 0.5 * z * z + d * (1.0 - v + log(v)):
                return d * v
    
    x = gamma_sample(alpha)
    y = gamma_sample(beta)
    
    if x + y < 1e-10:
        return alpha / (alpha + beta)
    
    return x / (x + y)


# ============================================================================
# MAIN CLASS - This is where all the magic happens
# ============================================================================
class MortyRescue:
    """
    The brain of the operation. This class:
    1. Tracks statistics for each planet (how many saved/lost)
    2. Decides which planet to use next
    3. Decides how many Morties to send
    4. Adapts as planet success rates change over time
    """
    
    def __init__(self, token: str):
        """Set up everything we need to track"""
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"}
        
        # For each planet, track:
        self.planet_stats = {
            planet: {
                "alpha": PRIOR_A,              # Successes (starts at 0.5)
                "beta": PRIOR_B,               # Failures (starts at 0.5)
                "trips": 0,                    # How many times we've used this planet
                "morties_saved": 0,            # Total Morties saved
                "morties_lost": 0,             # Total Morties lost
                "recent_outcomes": [],         # Last 10 results (1=saved, 0=lost)
                "last_trip_step": -999,        # When we last used this planet
                "consecutive_failures": 0,     # Failures in a row
                "consecutive_successes": 0,    # Successes in a row
                "size_hist": {1: 0, 2: 0, 3: 0},  # How often we sent 1, 2, or 3 Morties
            }
            for planet in PLANETS
        }
        
        # Overall tracking
        self.total_steps = 0                   # Total trips taken
        self.locked_planet = None              # Which planet we're locked to (None = not locked)
        self.lock_expires_at = -1              # When the lock expires
        self.pick_locked = 0                   # How many times we picked because locked
        self.pick_ts = 0                       # How many times we used Thompson Sampling
        self.size_hist_overall = {1: 0, 2: 0, 3: 0}  # Overall group size distribution
        
        # Lock tracking (for analysis)
        self._last_unlock_reason = None
        self._last_lock_r5 = None
        self._last_lock_gap = None
        self._current_lock_start_step = None
        
        # Random seed for reproducibility
        self.seed = int(time.time() * 1000) % (2**31)
        random.seed(self.seed)

    def start_episode(self) -> Dict:
        """Tell the API we're starting a new rescue mission"""
        try:
            response = requests.post(
                f"{BASE_URL}/api/mortys/start/",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error starting episode: {e}")
            return {}

    def send_morties(self, planet: str, count: int) -> Dict:
        """
        Send Morties to a planet and update our statistics.
        
        This is where the learning happens:
        1. Send Morties to the chosen planet
        2. See if they survived or died
        3. Update our beliefs about that planet
        4. Gradually forget old data (exponential discounting)
        """
        try:
            payload = {"planet": PLANET_TO_INT[planet], "morty_count": count}
            response = requests.post(
                f"{BASE_URL}/api/mortys/portal/",
                headers=self.headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Update basic counters
            self.total_steps += 1
            stats = self.planet_stats[planet]
            stats["trips"] += 1
            stats["last_trip_step"] = self.total_steps
            
            # Figure out how many survived vs died
            survived_count = count if data["survived"] else 0
            failed_count = count - survived_count
            
            # Track streaks (for emergency reset logic)
            if data["survived"]:
                stats["consecutive_failures"] = 0
                stats["consecutive_successes"] += 1
            else:
                stats["consecutive_failures"] += 1
                stats["consecutive_successes"] = 0
            
            # EMERGENCY RESET: If locked planet fails twice in a row, bail out!
            # This prevents us from riding a dying planet into the ground
            if (planet == self.locked_planet and 
                self.total_steps < self.lock_expires_at and 
                not data["survived"] and
                stats["consecutive_failures"] >= 2):
                # Reset this planet's stats and unlock
                stats["alpha"] = PRIOR_A
                stats["beta"] = PRIOR_B
                stats["consecutive_failures"] = 0
                stats["consecutive_successes"] = 0
                self.locked_planet = None
                self.lock_expires_at = -1
            
            # EXPONENTIAL DISCOUNTING: Gradually forget old data
            # This helps us adapt when planet success rates change
            # Multiply all alpha/beta by GAMMA (0.995) each step
            for p in PLANETS:
                p_stats = self.planet_stats[p]
                p_stats["alpha"] *= GAMMA  # Forget 0.5% of old successes
                p_stats["beta"] *= GAMMA   # Forget 0.5% of old failures
                
                # Make sure we don't decay to nothing
                current_ess = p_stats["alpha"] + p_stats["beta"]
                if current_ess < MIN_ESS:
                    scale = MIN_ESS / current_ess
                    p_stats["alpha"] *= scale
                    p_stats["beta"] *= scale
            
            # UPDATE BELIEFS: Add the new data we just learned
            stats["alpha"] += survived_count   # Add successes
            stats["beta"] += failed_count      # Add failures
            stats["morties_saved"] += survived_count
            stats["morties_lost"] += failed_count
            
            # Track last 10 outcomes (for calculating recent success rate)
            stats["recent_outcomes"].extend([1] * survived_count)
            stats["recent_outcomes"].extend([0] * failed_count)
            if len(stats["recent_outcomes"]) > 10:
                stats["recent_outcomes"] = stats["recent_outcomes"][-10:]
            
            return data
            
        except Exception as e:
            print(f"Error sending Morties: {e}")
            return {}

    def get_status(self) -> Dict:
        """Get current episode status"""
        try:
            response = requests.get(
                f"{BASE_URL}/api/mortys/status/",
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting status: {e}")
            return {}

    def get_recent_rate(self, planet: str, n: int = 5) -> float:
        """
        Calculate recent success rate for a planet.
        
        EXAMPLE: If last 5 outcomes were [1, 0, 1, 1, 1] (4 successes, 1 failure)
        Recent rate = 4/5 = 0.80 = 80% success rate
        """
        stats = self.planet_stats[planet]
        if len(stats["recent_outcomes"]) < n:
            n = len(stats["recent_outcomes"])
        if n == 0:
            return 0.0
        return sum(stats["recent_outcomes"][-n:]) / n

    def select_planet(self) -> str:
        """
        THE MOST IMPORTANT FUNCTION: Decide which planet to use.
        
        STRATEGY (in order):
        1. If we're locked to a planet, keep using it (unless it's dying)
        2. First 6 trips: Try each planet twice (exploration)
        3. After exploration: Try to lock onto the best planet
        4. If not locked: Use Thompson Sampling (random exploration)
        
        KEY INSIGHT: Lock early, stay locked, only bail if truly dying
        """
        
        # STEP 1: Check if our lock expired (after 60 trips)
        if self.locked_planet is not None and self.total_steps >= self.lock_expires_at:
            self._last_unlock_reason = "expired"
            self.locked_planet = None
            self.lock_expires_at = -1

        # STEP 2: If we're locked, stay locked (unless it's dying)
        if (self.locked_planet is not None and
            self.total_steps < self.lock_expires_at):
            # Check recent performance (last 5 outcomes)
            recent_rate = self.get_recent_rate(self.locked_planet, 5)
            
            # Only unlock if recent success < 20% (PATIENT!)
            # This is key: don't panic on 1-2 bad outcomes
            if recent_rate < 0.20:
                self._last_unlock_reason = f"cold_r5={recent_rate:.2f}"
                self.locked_planet = None
                self.lock_expires_at = -1
            else:
                # Still locked, keep using this planet
                self.pick_locked += 1
                return self.locked_planet
        
        # STEP 3: Initial exploration (first 6 trips)
        # Visit each planet twice to get initial data
        if self.total_steps < 6:
            for planet in PLANETS:
                if self.planet_stats[planet]["trips"] < 2:
                    return planet
        
        # STEP 4: Try to lock onto a winner
        # After exploration, look for a clear winner and commit
        if self.total_steps >= 6:
            # Calculate recent success rate for each planet
            recent_rates = {p: self.get_recent_rate(p, 5) for p in PLANETS}
            best_planet = max(PLANETS, key=lambda p: recent_rates[p])
            worst_planet = min(PLANETS, key=lambda p: recent_rates[p])
            
            gap = recent_rates[best_planet] - recent_rates[worst_planet]
            best_rate = recent_rates[best_planet]
            
            # Lock if: gap≥8%, rate≥50%, and 98% random chance
            # These are AGGRESSIVE thresholds = lock early!
            if (gap >= WINNER_GAP and
                best_rate >= WINNER_MIN_RATE and
                random.random() < WINNER_STICK):
                # Lock it in for 60 trips!
                self._last_lock_r5 = best_rate
                self._last_lock_gap = gap
                self._current_lock_start_step = self.total_steps
                self.locked_planet = best_planet
                self.lock_expires_at = self.total_steps + LOCK_DURATION
                return best_planet

        # STEP 5: Thompson Sampling (when not locked)
        # Sample from each planet's Beta distribution, pick the best sample
        # This adds randomness that helps us explore uncertain options
        samples = {}
        for planet in PLANETS:
            stats = self.planet_stats[planet]
            samples[planet] = beta_sample(stats["alpha"], stats["beta"])
        self.pick_ts += 1
        return max(samples, key=samples.get)

    def select_group_size(self, remaining: int, planet: str) -> int:
        """
        Decide how many Morties to send at once.
        
        STRATEGY:
        - Send 1 when uncertain (gathering data)
        - Send 2 when moderately confident (58-88% success)
        - Send 3 when very confident (88%+ success)
        
        WHY THIS MATTERS:
        - Sending more = faster progress when planet is good
        - Sending more = bigger losses when planet is bad
        - Balance speed vs safety based on confidence
        """
        stats = self.planet_stats[planet]
        
        # Always start with size 1 for first 3 trips (gathering data)
        if stats["trips"] < 3:
            return 1
        
        # Get recent success rates
        recent_5 = self.get_recent_rate(planet, 5)  # Last 5 outcomes
        recent_3 = self.get_recent_rate(planet, 3)  # Last 3 outcomes
        
        # If we're LOCKED to this planet, be more aggressive
        # Use recent_3 (more reactive to current conditions)
        if (planet == self.locked_planet and 
            self.total_steps < self.lock_expires_at):
            if recent_3 >= SUPER_HOT_THRESHOLD:    # 88%+ → send 3
                return min(3, remaining)
            elif recent_3 >= BLAZING_THRESHOLD:    # 78%+ → send 3
                return min(3, remaining)
            elif recent_3 >= VERY_HOT_THRESHOLD:   # 68%+ → send 2
                return min(2, remaining)
            else:                                   # < 68% → send 1
                return 1
        
        # If NOT locked, be more conservative
        # Use recent_5 (more stable, less reactive)
        if recent_5 >= SUPER_HOT_THRESHOLD:        # 88%+ → send 3
            return min(3, remaining)
        elif recent_5 >= BLAZING_THRESHOLD:        # 78%+ → send 3
            return min(3, remaining)
        elif recent_5 >= VERY_HOT_THRESHOLD:       # 68%+ → send 2
            return min(2, remaining)
        elif recent_5 >= HOT_THRESHOLD:            # 58%+ → send 2
            return min(2, remaining)
        elif recent_5 < COLD_THRESHOLD:            # < 30% → send 1
            return 1
        else:                                       # 30-58% → send 1
            return 1


    def run_rescue_mission(self):
        """
        Run the full rescue mission: send all 1000 Morties.
        
        THE MAIN LOOP:
        1. Pick a planet (using select_planet)
        2. Pick how many to send (using select_group_size)
        3. Send them and see what happens
        4. Update our beliefs
        5. Repeat until all Morties are sent
        """
        status = self.start_episode()
        if not status:
            return None

        remaining = status["morties_in_citadel"]
        saved = status["morties_on_planet_jessica"]

        # Track lock/unlock events (for analysis)
        lock_events = []
        unlock_events = []
        
        # MAIN LOOP: Keep going until all Morties are sent
        while remaining > 0:
            prev_locked = self.locked_planet
            planet = self.select_planet()

            # Track lock/unlock events
            if prev_locked is None and self.locked_planet is not None:
                lock_events.append({
                    'step': self.total_steps,
                    'planet': self.locked_planet,
                    'duration': self.lock_expires_at - self.total_steps,
                    'r5_at_lock': getattr(self, '_last_lock_r5', None),
                    'gap_at_lock': getattr(self, '_last_lock_gap', None)
                })
                self._last_lock_r5 = None
                self._last_lock_gap = None

            elif prev_locked is not None and self.locked_planet is None:
                actual_duration = self.total_steps - self._current_lock_start_step if self._current_lock_start_step else 0
                unlock_events.append({
                    'step': self.total_steps,
                    'planet': prev_locked,
                    'reason': getattr(self, '_last_unlock_reason', 'unlocked'),
                    'actual_duration': actual_duration
                })
                self._last_unlock_reason = None
                self._current_lock_start_step = None

            group_size = self.select_group_size(remaining, planet)
            self.size_hist_overall[group_size] += 1
            self.planet_stats[planet]['size_hist'][group_size] += 1

            result = self.send_morties(planet, group_size)
            if not result:
                status = self.get_status()
                if status:
                    remaining = status.get("morties_in_citadel", remaining)
                    saved = status.get("morties_on_planet_jessica", saved)
                time.sleep(0.5)
                continue
            
            remaining = result["morties_in_citadel"]
            saved = result["morties_on_planet_jessica"]
            lost = result["morties_lost"]
            
            if SLOW_MODE:
                time.sleep(0.3)
        
        # Final results
        final_status = self.get_status()
        if final_status:
            success_rate = final_status["morties_on_planet_jessica"] / 1000
            print(f"Run complete: {final_status['morties_on_planet_jessica']}/1000 ({success_rate:.1%})")

        # Return comprehensive run statistics
        if not final_status:
            return None

        # Compute overall average group size
        total_morties = sum(s["morties_saved"] + s["morties_lost"] for s in self.planet_stats.values())
        total_trips = sum(s["trips"] for s in self.planet_stats.values())
        avg_group_size_overall = total_morties / total_trips if total_trips > 0 else 0

        run_stats = {
            'success_rate': final_status['morties_on_planet_jessica'] / 1000,
            'morties_saved': final_status['morties_on_planet_jessica'],
            'morties_lost': final_status['morties_lost'],
            'total_steps': final_status['steps_taken'],
            'seed': self.seed,
            'lock_events': lock_events,
            'unlock_events': unlock_events,
            'pick_locked': self.pick_locked,
            'pick_ts': self.pick_ts,
            'avg_group_size_overall': avg_group_size_overall,
            'size_hist_overall': self.size_hist_overall,
            'planets': {}
        }

        # Planet-level stats
        for planet in PLANETS:
            s = self.planet_stats[planet]
            total_morties = s["morties_saved"] + s["morties_lost"]
            avg_size = total_morties / s["trips"] if s["trips"] > 0 else 0
            planet_rate = s["morties_saved"] / total_morties if total_morties > 0 else 0

            run_stats['planets'][planet] = {
                'trips': s['trips'],
                'morties_saved': s['morties_saved'],
                'morties_lost': s['morties_lost'],
                'avg_size': avg_size,
                'success_rate': planet_rate,
                'alpha': s['alpha'],
                'beta': s['beta'],
                'size_hist': s['size_hist']
            }

        return run_stats


# ============================================================================
# MAIN ENTRY POINT - Run the rescue mission
# ============================================================================
def main():
    """
    Run a single rescue mission and print the results.
    
    This is what happens when you run: python morty_rescue.py
    """
    print("Morty Rescue - Starting mission...")

    token = API_TOKEN
    if not token:
        print("No API token set!")
        return

    # Create the rescue object and run the mission
    rescue = MortyRescue(token)
    run_stats = rescue.run_rescue_mission()

    # Print final results
    if run_stats:
        print(f"\n{'='*60}")
        print(f"MISSION COMPLETE")
        print(f"{'='*60}")
        print(f"Success Rate: {run_stats['success_rate']:.1%}")
        print(f"Morties Saved: {run_stats['morties_saved']}/1000")
        print(f"Total Steps: {run_stats['total_steps']}")
        print(f"Locks Used: {len(run_stats['lock_events'])}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()

