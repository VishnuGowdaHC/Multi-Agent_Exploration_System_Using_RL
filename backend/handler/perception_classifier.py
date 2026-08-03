import yaml
import logging

class PerceptionClassifier:
    def __init__(self, config_path="../config/threats/jungle_demo.yml"):
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

                # Fixed: Handle direct root-level profiles or nested 'threats' key gracefully
                if config_data:
                    if isinstance(config_data, dict):
                        if 'threats' in config_data:
                            self.threat_profiles = config_data['threats']
                        else:
                            # Treat root level keys as profiles since your YAML has tiger/deer/wasp directly at top-level
                            self.threat_profiles = {k: v for k, v in config_data.items() if k != 'default'}

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