import builtins
import json
import socket

import pytest
import torch
from safetensors.torch import save_file
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, processors

from litgpt.gliclass_local import LocalGLiClass, LocalGLiClassPipeline


@pytest.fixture
def tiny_detector(tmp_path):
    config = dict(architecture_type="uni-encoder", scorer_type="mlp", pooling_strategy="first",
                  class_token_pooling="average", embed_class_token=True, normalize_features=False,
                  extract_text_features=False, use_segment_embeddings=True, projector_hidden_act="gelu",
                  encoder_layer_id=-1, hidden_size=16, logit_scale_init_value=2.6592,
                  class_token_index=50, text_token_index=51, example_token_index=52,
                  pad_token_id=0, prompt_first=True,
                  encoder_config=dict(model_type="modernbert", hidden_size=16, num_hidden_layers=2,
                                      num_attention_heads=2, intermediate_size=24, vocab_size=64,
                                      max_position_embeddings=256, local_attention=16, global_attn_every_n_layers=2,
                                      pad_token_id=0, cls_token_id=1, sep_token_id=2))
    torch.manual_seed(3)
    model = LocalGLiClass(config)
    save_file({"model." + name: value for name, value in model.state_dict().items()},
              str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(config))
    vocab = {f"word{i}": i for i in range(64)}
    for token, index in (("[PAD]", 0), ("[CLS]", 1), ("[SEP]", 2), ("[UNK]", 3),
                         ("<<LABEL>>", 50), ("<<SEP>>", 51), ("<<EXAMPLE>>", 52)):
        vocab.pop(f"word{index}")
        vocab[token] = index
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in ("<<LABEL>>", "<<SEP>>")])
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
                                                               special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    return tmp_path


def test_native_loader_needs_no_gliclass_package_or_network_and_preserves_padding(tiny_detector, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gliclass" or name.startswith("gliclass."):
            raise AssertionError("native adapter must not import the external GLiClass package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket.socket, "connect", lambda *args: pytest.fail("unexpected network access"))
    pipeline = LocalGLiClassPipeline(tiny_detector, max_length=128)
    texts = ["word5", "word6 word7 word8", "word9 word10"]
    labels = ["word11", "word12"]
    batched = pipeline(texts, labels, batch_size=3)
    single = pipeline(texts, labels, batch_size=1)
    as_tensor = lambda rows: torch.tensor([[item["score"] for item in row] for row in rows])
    torch.testing.assert_close(as_tensor(batched), as_tensor(single), atol=1e-6, rtol=1e-5)
    assert torch.isfinite(as_tensor(batched)).all()
    assert not pipeline.model.training and not any(p.requires_grad for p in pipeline.model.parameters())


def test_native_loader_refuses_missing_weights_instead_of_initializing_randomly(tiny_detector):
    (tiny_detector / "model.safetensors").unlink()
    save_file({"model.logit_scale": torch.tensor(0.)}, str(tiny_detector / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Missing key"):
        LocalGLiClassPipeline(tiny_detector)


def test_native_loader_refuses_unsupported_checkpoint_semantics(tiny_detector):
    path = tiny_detector / "config.json"
    config = json.loads(path.read_text())
    config["normalize_features"] = True
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="normalize_features"):
        LocalGLiClassPipeline(tiny_detector)
