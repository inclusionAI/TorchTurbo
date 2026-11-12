"""GLM templates follow LlamaFactory's history-CoT training contract."""

from copy import deepcopy

import pytest

pytest.importorskip("llamafactory")

from llamafactory.data.template import TEMPLATES
from torchturbo.models.glm5_3_flash.lf_template import register


class CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))


@pytest.mark.parametrize("discard_history", [False, True])
def test_glm_template_preserves_last_cot_and_optionally_discards_history(discard_history):
    register()
    template = TEMPLATES["glm5_3_flash"]
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "<think>old reasoning</think>first answer"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "<think>last reasoning</think>last answer"},
    ]
    original = deepcopy(messages)
    pairs = template.encode_multiturn(
        CharacterTokenizer(), messages, None, None, discard_history,
    )
    first_answer = "".join(map(chr, pairs[0][1]))
    last_answer = "".join(map(chr, pairs[-1][1]))
    assert ("old reasoning" in first_answer) is not discard_history
    assert "last reasoning" in last_answer
    assert template.add_thought() == "</think>"
    assert messages == original
