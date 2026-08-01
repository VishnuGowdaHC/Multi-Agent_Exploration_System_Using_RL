import yaml
import logging

class PerceptionClassifier:
    def __init__(self, config_path=""):
        self.config_path = config_path
        self.threat_profiles = {}
        self.default_profile = {
            "lethality": 1.0,
            "radius": 1.0,
            "persistence": "static",
            "detectability": 1.0
        }
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                config_data = yaml.safe_load(file)

                if config_data and 'threats' in config_data:
                    self.threat_profiles = config_data['threats']

                if config_data and 'default' in config_data:
                    self.default_profile = config_data['default']
            print(f"Loaded threat profiles: {self.threat_profiles}")

        except FileNotFoundError:
            logging.error(f"Configuration file {self.config_path} not found. Defaulting to safe baselines.")
        except yaml.YAMLError as exc:
            logging.error(f"Error parsing YAML configuration: {exc}")

    def classify(self, unity_tag):
        normalized_tag = str(unity_tag).strip().lower()

        return self.threat_profiles.get(normalized_tag, self.default_profile)