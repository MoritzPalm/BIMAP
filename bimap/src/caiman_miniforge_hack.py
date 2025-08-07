import subprocess

def run_in_caiman(env_name, work_dir, script_path, *args):
    activate_bat = r"C:\Users\morit\Miniforge3\Scripts\activate.bat"
    cmd = (
        f'cd /d "{work_dir}" '
        f'&& CALL "{activate_bat}" "{env_name}" '
        f'&& python "{script_path}" {" ".join(args)}'
    )
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Error:\n{result.stderr}")
    return result.stdout


if __name__ == '__main__':
    out = run_in_caiman("caiman", r"C:\Users\morit\caiman_data\demos\notebooks",
                        "mc_normcorre.py", "C:/Users/morit/Documents/Studium/BIMAP/data/input/strong_movement/b5czi.tif")
    print(out)