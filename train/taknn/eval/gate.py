from .elo import MatchRunner
from .tactical import TacticalSuiteRunner

class PromotionGate:
    def __init__(self, current_best_model, tactical_suite_path, size=6):
        self.current_best = current_best_model
        self.tactical_suite_path = tactical_suite_path
        self.size = size

    def check_promotion(self, candidate_model, elo_target=50, tactical_regress_limit=0.05):
        # 1. Check Elo match
        runner = MatchRunner(candidate_model, self.current_best, size=self.size, games=200)
        elo_diff, ci = runner.run()
        
        if elo_diff - ci < elo_target:
            print(f"Promotion failed: Elo difference {elo_diff:.1f} ± {ci:.1f} is not clearly above {elo_target}")
            return False
            
        # 2. Check tactical suite
        tactical_runner = TacticalSuiteRunner(candidate_model, self.tactical_suite_path)
        candidate_tactical = tactical_runner.run()['accuracy']
        
        best_tactical_runner = TacticalSuiteRunner(self.current_best, self.tactical_suite_path)
        best_tactical = best_tactical_runner.run()['accuracy']
        
        if candidate_tactical < best_tactical - tactical_regress_limit:
            print(f"Promotion failed: Tactical accuracy {candidate_tactical:.1%} regressed from {best_tactical:.1%}")
            return False
            
        # 3. Check browser latency (placeholder)
        # perf_gate = BrowserPerfGate(candidate_model)
        # if not perf_gate.check():
        #     return False
            
        print(f"Promotion successful: Elo +{elo_diff:.1f}, Tactical {candidate_tactical:.1%}")
        return True
