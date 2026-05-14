class MeanReversionStrategy:
    def __init__(self, z_threshold=2.2):
        self.z_threshold = z_threshold

    def check(self, price, vwap, rsi):
        # Calculate Deviation
        z_score = (price - vwap) / (vwap * 0.001)
        
        if z_score < -self.z_threshold and rsi < 30:
            return "CALL", f"regime=MEAN_REVERT | score={abs(z_score):.2f}/5.0 | rsi={rsi}"
        elif z_score > self.z_threshold and rsi > 70:
            return "PUT", f"regime=MEAN_REVERT | score={abs(z_score):.2f}/5.0 | rsi={rsi}"
        
        return "NONE", "NONE"

class VolatilityStrategy:
    def __init__(self, threshold=0.45):
        self.threshold = threshold

    def check(self, obi, velocity):
        if obi > self.threshold and velocity > 0:
            return "CALL", f"regime=VOL_SCALP | score={obi:.2f}/0.5 | obi={obi}"
        elif obi < -self.threshold and velocity < 0:
            return "PUT", f"regime=VOL_SCALP | score={abs(obi):.2f}/0.5 | obi={obi}"
        return "NONE", "NONE"