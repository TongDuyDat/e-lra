import subprocess
import os
from datetime import datetime
import time


# Định nghĩa các kịch bản huấn luyện và test
scenarios = [
    (["data/configs/kvasir-seg.py"], "data/configs/CVC-ColonDB.py", "kvasir-seg_to_CVC-ColonDB"),
    (["data/configs/kvasir-seg.py"], "data/configs/ETIS-LaribPolyDB.py", "kvasir-seg_to_ETIS-LaribPolyDB"),
    (["data/configs/CVC-ColonDB.py"], "data/configs/kvasir-seg.py", "CVC-ColonDB_to_kvasir-seg"),
    (["data/configs/CVC-ColonDB.py"], "data/configs/ETIS-LaribPolyDB.py", "CVC-ColonDB_to_ETIS-LaribPolyDB"),
    (["data/configs/ETIS-LaribPolyDB.py"], "data/configs/kvasir-seg.py", "ETIS-LaribPolyDB_to_kvasir-seg"),
    (["data/configs/ETIS-LaribPolyDB.py"], "data/configs/CVC-ColonDB.py", "ETIS-LaribPolyDB_to_CVC-ColonDB"),
    (["data/configs/kvasir-seg.py", "data/configs/CVC-ColonDB.py"], "data/configs/ETIS-LaribPolyDB.py", "kvasir-seg_CVC-ColonDB_to_ETIS-LaribPolyDB"),
    (["data/configs/CVC-ColonDB.py", "data/configs/ETIS-LaribPolyDB.py"], "data/configs/kvasir-seg.py", "CVC-ColonDB_ETIS-LaribPolyDB_to_kvasir-seg"),
    (["data/configs/kvasir-seg.py", "data/configs/ETIS-LaribPolyDB.py"], "data/configs/CVC-ColonDB.py", "kvasir-seg_ETIS-LaribPolyDB_to_CVC-ColonDB")
]

# Tham số chung
common_args = [
    "--config", "configs/train_config.py",
    "--batch-size", "16"
]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# Hàm chạy một kịch bản tuần tự
def run_scenario(train_data, test_data, names):
    output_dir = f"logs/{timestamp}/{names}"
    os.makedirs(output_dir, exist_ok=True)

    # Chuẩn bị lệnh gọi subprocess
    cmd = ["python", "train.py"]
    cmd.extend(["--train-data"] + train_data)
    cmd.extend(["--test-data", test_data])
    cmd.extend(['--names', f'{names}'])
    cmd.extend(common_args)
    # Ghi log
    print(f"[{time.ctime()}] Starting scenario {names}: Train on {','.join(train_data)} | Test on {test_data}")
    with open(f"{output_dir}/log.txt", "a") as log_file:
        log_file.write(f"[{time.ctime()}] Command: {' '.join(cmd)}\n")

    # Chạy subprocess và chờ hoàn thành
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with open(f"{output_dir}/log.txt", "a") as log_file:
        log_file.write(f"[{time.ctime()}] STDOUT:\n{result.stdout}\n")
        log_file.write(f"[{time.ctime()}] STDERR:\n{result.stderr}\n")
        log_file.write(f"[{time.ctime()}] Return code: {result.returncode}\n")

    print(f"[{time.ctime()}] Completed scenario {names}")

# Chạy các kịch bản tuần tự
for idx, (train_data, test_data, names) in enumerate(scenarios):
    run_scenario(train_data, test_data, names)

print(f"[{time.ctime()}] All scenarios completed at {datetime.now()}")