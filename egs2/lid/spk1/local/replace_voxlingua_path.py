voxlingua_wavscp = "/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/dump/raw/train_voxlingua107_lang/wav.scp"
voxlingua_wavscp_replace_path = "/ocean/projects/cis210027p/qwang20/espnet/egs2/lid/spk1/dump/raw/train_voxlingua107_lang/wav.scp.replace_path"

voxlingua_wavscp_replace_path_dump = []
with open(voxlingua_wavscp, "r") as f:
    for line in f:
        utt_id, path = line.strip().split()
        # Replace the "downloads" tp "voxlingua_107" in path
        path = path.replace("downloads", "voxlingua_107")
        voxlingua_wavscp_replace_path_dump.append(f"{utt_id} {path}\n")

with open(voxlingua_wavscp_replace_path, "w") as f:
    f.writelines(voxlingua_wavscp_replace_path_dump)
print(f"Replace the path in {voxlingua_wavscp} and save to {voxlingua_wavscp_replace_path}")