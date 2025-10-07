import json
from pathlib import Path

data_file = Path(__file__).parent / "position_data.json"

data = json.loads(data_file.read_text())
for data_point in data["tracking_data"]:
    time = data_point["timestamp"]
    distance = data_point["position_x_cm"]
    print(time, ",", distance - 1)
