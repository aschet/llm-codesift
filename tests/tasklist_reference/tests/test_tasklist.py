"""The tests a model is expected to write for itself, written here instead.

Written with pytest, like the other reference solution and unlike the rest of
this repository, because the grader runs a model's tests by shelling out to
pytest and models write pytest.
"""
import pytest

import tasklist as tl


@pytest.fixture()
def store():
    s = tl.new_store()
    tl.add(s, "Buy milk", "Semi-skimmed", "#3366cc")
    tl.add(s, "Write report", "Quarterly summary", "#cc7722")
    return s


def test_a_new_task_is_open_and_last(store):
    made = tl.add(store, "Third")
    assert made["done"] is False
    assert tl.tasks(store)[-1]["id"] == made["id"]


def test_an_empty_title_is_refused(store):
    with pytest.raises(ValueError):
        tl.add(store, "")


def test_update_leaves_unnamed_fields_alone(store):
    first = tl.tasks(store)[0]
    after = tl.update(store, first["id"], done=True)
    assert after["done"] is True and after["description"] == "Semi-skimmed"


def test_update_refuses_a_field_it_does_not_own(store):
    with pytest.raises(ValueError):
        tl.update(store, tl.tasks(store)[0]["id"], position=9)


def test_a_deleted_task_is_gone(store):
    first = tl.tasks(store)[0]
    assert tl.delete(store, first["id"]) is True
    assert tl.get(store, first["id"]) is None
    assert tl.delete(store, first["id"]) is False


def test_filters_narrow_the_listing(store):
    first = tl.tasks(store)[0]
    tl.update(store, first["id"], done=True)
    assert [t["id"] for t in tl.tasks(store, done=True)] == [first["id"]]
    assert len(tl.tasks(store, q="QUARTERLY")) == 1


def test_reorder_refuses_an_unknown_id(store):
    with pytest.raises(ValueError):
        tl.reorder(store, [tl.tasks(store)[0]["id"], 4242])


def test_a_saved_store_comes_back_the_same(store, tmp_path):
    ids = [t["id"] for t in tl.tasks(store)][::-1]
    tl.reorder(store, ids)
    path = tmp_path / "s.json"
    tl.save(store, str(path))
    assert [t["id"] for t in tl.tasks(tl.load(str(path)))] == ids


def test_loading_a_missing_file_gives_an_empty_store(tmp_path):
    assert tl.tasks(tl.load(str(tmp_path / "absent.json"))) == []
