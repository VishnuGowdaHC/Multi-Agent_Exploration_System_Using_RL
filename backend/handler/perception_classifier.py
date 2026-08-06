import yaml
import logging
from pathlib import Path

class PerceptionClassifier:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "threats" / "jungle_demo.yml"
        
        self.config_path = config_path
        self.threat_profiles = {}
        self.default_profile = {
            "lethality": 1.0,
            "radius": 1.0,
            "persistence": "static",
        }
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)

                if config_data:
                    if isinstance(config_data, dict):
                        if 'threats' in config_data:
                            raw_profiles = config_data['threats']
                        else:
                            raw_profiles = {k: v for k, v in config_data.items() if k != 'default'}

                        normalized = {}
                        for name, profile in raw_profiles.items():
                            profile = dict(profile)
                            if 'radius_m' in profile and 'radius' not in profile:
                                profile['radius'] = profile.pop('radius_m')
                            normalized[name.lower()] = profile
                        self.threat_profiles = normalized

                        if 'default' in config_data:
                            self.default_profile = config_data['default']

                print(f"Loaded threat profiles: {list(self.threat_profiles.keys())}")

        except FileNotFoundError:
            logging.error(f"Configuration file {self.config_path} not found. Defaulting to safe baselines.")
        except yaml.YAMLError as exc:
            logging.error(f"Error parsing YAML configuration: {exc}")

    def classify(self, unity_tag):
        normalized_tag = str(unity_tag).strip().lower()
        return self.threat_profiles.get(normalized_tag, self.default_profile)