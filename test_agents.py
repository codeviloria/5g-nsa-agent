"""Offline tests: no Ollama needed. Scripted model responses exercise the full tool loop."""
import json

import agent_raw
from agent_raw import function_to_schema, run_agent
from nsa_tools import (TOOLS, audit_endc_config, check_nr_pci_conflicts, nr_link_budget,
                       ssb_beam_report)


def test_tools_scenario():
    a021 = audit_endc_config("BAQ_021_A")["findings"]
    assert [f["check"] for f in a021] == ["smtc_alignment"]
    assert a021[0]["fix"] == {"smtc_offset_ms": 10}
    a034 = audit_endc_config("BAQ_034_A")["findings"]
    assert [f["check"] for f in a034] == ["b1_threshold"]
    pci = check_nr_pci_conflicts("BAQ_034_N78_A")
    assert [i["type"] for i in pci["issues"]] == ["mod3"]
    assert ssb_beam_report("BAQ_034_N78_A")["weak_beams"] == [6, 7]
    lb = nr_link_budget(500)
    assert lb["passes_b1"] and lb["ss_rsrp_dbm"] == -94.1
    assert "error" in nr_link_budget(9000)


def test_schemas():
    s = function_to_schema(audit_endc_config)["function"]
    assert s["name"] == "audit_endc_config"
    assert s["parameters"]["required"] == ["anchor_cell_id"]
    lb = function_to_schema(nr_link_budget)["function"]["parameters"]
    assert lb["properties"]["distance_m"]["type"] == "number"
    assert lb["properties"]["bandwidth_mhz"]["type"] == "integer"
    assert lb["properties"]["los"]["type"] == "boolean"
    assert lb["required"] == ["distance_m"]
    assert "meters" in lb["properties"]["distance_m"]["description"]
    assert len(TOOLS) == 5


def test_raw_loop():
    script = iter([
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_endc_kpis", "arguments": {"anchor_cell_id": "BAQ_021_A"}}},
            {"function": {"name": "audit_endc_config", "arguments": {"anchor_cell_id": "BAQ_021_A"}}}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "ssb_beam_report", "arguments": {"nr_cell_id": "NOPE"}}}]},
        {"role": "assistant", "content": "SMTC offset 0 ms misses the SSB burst at 10 ms."},
    ])
    seen = []

    def fake_chat(messages, tools):
        seen.append([m for m in messages if m["role"] == "tool"])
        return next(script)

    agent_raw.chat = fake_chat
    out = run_agent("diagnose BAQ_021_A", verbose=False)
    assert "SMTC" in out
    tool_msgs = seen[-1]
    assert len(tool_msgs) == 3
    assert json.loads(tool_msgs[1]["content"])["findings"][0]["check"] == "smtc_alignment"
    assert "error" in json.loads(tool_msgs[2]["content"])     # bad cell id goes back as data


def test_langchain_agent():
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage

    from agent_langchain import build_agent

    class FakeToolModel(GenericFakeChatModel):
        def bind_tools(self, tools, **kw):
            return self

    msgs = iter([
        AIMessage(content="", tool_calls=[{"name": "check_nr_pci_conflicts", "id": "1",
                                           "args": {"nr_cell_id": "BAQ_034_N78_A"}}]),
        AIMessage(content="PCI mod-3 conflict with BAQ_021_N78_A."),
    ])
    agent = build_agent(FakeToolModel(messages=msgs))
    res = agent.invoke({"messages": [{"role": "user", "content": "SCG failures on BAQ_034_A?"}]})
    tool_msg = [m for m in res["messages"] if m.type == "tool"][0]
    assert json.loads(tool_msg.content)["issues"][0]["type"] == "mod3"
    assert "mod-3" in res["messages"][-1].content


if __name__ == "__main__":
    for t in (test_tools_scenario, test_schemas, test_raw_loop, test_langchain_agent):
        t()
        print("PASS", t.__name__)
