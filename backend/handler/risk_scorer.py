class RiskScorer:
    def calculate_immediate_risk(self, feature_vector, distance):
        lethality = feature_vector.get("lethality", 0.0)
        radius = feature_vector.get("radius", 1.0)

        if distance>radius:
            return 0.0

        safe_dist = max(float(distance), 1.0)

        risk = lethality * (1.0/(safe_dist**2)) * radius

        return risk

    def calculate_cell_risk(self, cx, cz, active_threats):
        total_risk = 0.0

        for threat in active_threats:
            tx, tz = threat['pos']
            features = threat['features']

            dist = ((cx - tx) ** 2 + (cz - tz) ** 2) ** 0.5

            if dist <= features['radius']:
                safe_dist = max(dist, 1.0)
                total_risk += features['lethality'] * (1.0 / (safe_dist ** 2)) * features['radius']

        return total_risk