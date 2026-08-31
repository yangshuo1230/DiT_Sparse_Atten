import types

from profiles.provenance import collect


def test_collect_provenance_records_config_and_revisions(tmp_path):
    args = types.SimpleNamespace(steps=5, output=tmp_path / "video.mp4")
    result = collect(args, tmp_path)
    assert result["command_config"]["steps"] == 5
    assert result["command_config"]["output"].endswith("video.mp4")
    assert result["environment"]["study_git"]["commit"]
    assert result["environment"]["wan_git"] is None
