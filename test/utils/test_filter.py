"""Tests for resolve_filter_ids utility."""

from visionframework.utils.filter import resolve_filter_ids


class TestResolveFilterIds:
    def test_none_input(self):
        assert resolve_filter_ids(None, None) is None

    def test_empty_list(self):
        assert resolve_filter_ids([], None) is None

    def test_int_ids(self):
        result = resolve_filter_ids([0, 2], None)
        assert result == {0, 2}

    def test_str_names_list(self):
        names = ["person", "car", "bus"]
        result = resolve_filter_ids(["person", "bus"], names)
        assert result == {0, 2}

    def test_str_names_dict(self):
        names = {0: "person", 2: "car", 5: "bus"}
        result = resolve_filter_ids(["person", "bus"], names)
        assert result == {0, 5}

    def test_mixed_int_str(self):
        names = ["person", "car", "bus"]
        result = resolve_filter_ids([0, "bus"], names)
        assert result == {0, 2}

    def test_case_insensitive(self):
        names = ["Person", "Car"]
        result = resolve_filter_ids(["person", "CAR"], names)
        assert result == {0, 1}

    def test_unknown_name_ignored(self):
        names = ["person", "car"]
        result = resolve_filter_ids(["person", "unknown_class"], names)
        assert result == {0}

    def test_all_unknown_returns_none(self):
        names = ["person", "car"]
        result = resolve_filter_ids(["unknown"], names)
        assert result is None

    def test_no_class_names(self):
        result = resolve_filter_ids([0, 1], None)
        assert result == {0, 1}
