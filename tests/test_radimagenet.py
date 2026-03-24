import importlib.util
import json
import sys
import types
from enum import Enum
from pathlib import Path


class _StubAgent:
    def __init__(self, *args, **kwargs):
        pass

    def instructions(self, fn):
        return fn


class _StubFieldInfo:
    def __init__(self, default=None, default_factory=None):
        self.default = default
        self.default_factory = default_factory


def _field(default=None, default_factory=None):
    return _StubFieldInfo(default=default, default_factory=default_factory)


class _BaseModel:
    def __init__(self, **kwargs):
        for name, value in self.__class__.__dict__.items():
            if name.startswith("_") or callable(value):
                continue
            if name in kwargs:
                setattr(self, name, kwargs[name])
                continue
            if isinstance(value, _StubFieldInfo):
                if value.default_factory is not None:
                    setattr(self, name, value.default_factory())
                else:
                    setattr(self, name, value.default)
            else:
                setattr(self, name, value)

        for name, value in kwargs.items():
            if not hasattr(self, name):
                setattr(self, name, value)

    def model_dump(self):
        out = {}
        for name in self.__annotations__:
            value = getattr(self, name, None)
            if isinstance(value, list):
                out[name] = [v.value if isinstance(v, Enum) else v for v in value]
            elif isinstance(value, Enum):
                out[name] = value.value
            else:
                out[name] = value
        return out

    def model_dump_json(self, indent=None):
        return json.dumps(self.model_dump(), indent=indent)


def _install_dependency_stubs():
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda: None
    sys.modules["dotenv"] = dotenv

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = lambda iterable: iterable
    sys.modules["tqdm"] = tqdm_mod

    bio_mod = types.ModuleType("Bio")
    entrez = types.SimpleNamespace(email=None)
    bio_mod.Entrez = entrez
    sys.modules["Bio"] = bio_mod

    pydantic_ai_mod = types.ModuleType("pydantic_ai")
    pydantic_ai_mod.Agent = _StubAgent
    pydantic_ai_mod.RunContext = type("RunContext", (), {"__class_getitem__": classmethod(lambda cls, item: cls)})
    sys.modules["pydantic_ai"] = pydantic_ai_mod

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.BaseModel = _BaseModel
    pydantic_mod.Field = _field
    sys.modules["pydantic"] = pydantic_mod


def load_build_module(monkeypatch):
    monkeypatch.setenv("ENTREZ_EMAIL", "unit-test@example.com")
    _install_dependency_stubs()

    module_name = "build_database_table_under_test"
    module_path = Path(__file__).resolve().parents[1] / "src" / "build_database_table.py"

    if module_name in sys.modules:
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def test_radimagenet_seed_text_contains_expected_values(monkeypatch):
    module = load_build_module(monkeypatch)

    title, abstract, link = module.get_radimagenet_seed_text()

    assert title == module.RADIMAGENET_TITLE
    assert "RadImageNet" in abstract
    assert "1.35 million annotated medical images" in abstract
    assert link == "https://doi.org/10.1148/ryai.210315"


def test_serialize_dataset_output_produces_valid_pretty_json(monkeypatch):
    module = load_build_module(monkeypatch)

    title, abstract, link = module.get_radimagenet_seed_text()
    dataset = module.RadiologyDataset(
        name="RadImageNet database",
        num_patients=131872,
        modalities=[module.Modality.CT, module.Modality.MRI, module.Modality.US],
        paper_title=title,
        paper_abstract=abstract,
        paper_link=link,
    )

    output = module.serialize_dataset_output(dataset)
    parsed = json.loads(output)

    assert parsed["name"] == "RadImageNet database"
    assert parsed["paper_link"] == "https://doi.org/10.1148/ryai.210315"
    assert parsed["modalities"] == ["CT", "MRI", "US"]
    assert output.startswith("{\n")
    assert "\n    \"name\": \"RadImageNet database\"" in output
